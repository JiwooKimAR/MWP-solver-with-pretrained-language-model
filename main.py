import os
import sys
import torch
import argparse

from collections import OrderedDict
from dataloader import Dataset
from evaluation import Evaluator
from experiment import EarlyStop, train_model
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main_train_test(argv):
    # multiprocessing.set_start_method('spawn')

    # read configs
    config = Config(main_conf_path='./', model_conf_path='model_config')

    # apply system arguments if exist
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = ' '.join(sys.argv[1:]).split(' ')
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip('-')
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    gpu = config.get_param('Experiment', 'gpu')
    gpu = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_param('Experiment', 'model_name')

    # set seed
    seed = config.get_param('Experiment', 'seed')
    set_random_seed(seed)

    # logger
    log_dir = make_log_dir(os.path.join('saves', model_name))
    logger = Logger(log_dir)
    config.save(log_dir)

    # dataset
    dataset_name = config.get_param('Dataset', 'dataset')
    dataset = Dataset(model_name, **config['Dataset'])

    # early stop
    early_stop = EarlyStop(**config['EarlyStop'])

    # evaluator()
    evaluator = Evaluator(early_stop.early_stop_measure, **config['Evaluator'])

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    import model

    MODEL_CLASS = getattr(model, model_name)

    # build model
    model = MODEL_CLASS(dataset, config['Model'], device)
    model.logger = logger
    
    ################################## TRAIN & PREDICT
    # train
    try:
        valid_score, train_time = train_model(model, dataset, evaluator, early_stop, logger, config)
    except (KeyboardInterrupt, SystemExit):
        valid_score, train_time = dict(), 0
        logger.info("학습을 중단하셨습니다.")

    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

    # test
    model.eval()
    model.restore(logger.log_dir)
    test_score = dict()
    for testset in dataset.testsets:
        test_score.update(evaluator.evaluate(model, dataset, testset))

    # show result
    evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
    evaluation_table.add_row('Valid', valid_score)
    evaluation_table.add_row('Test', test_score)

    # evaluation_table.show()
    logger.info(evaluation_table.to_string())
        
    logger.info("Saved to %s" % (log_dir))

def main_submit(args):
    # read configs
    config = Config(main_conf_path=args.path, model_conf_path=args.path)

    # Final test set (dataset/problemsheet.json)
    config.main_config['Dataset']['dataset'] = '/home/agc2021/dataset'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_param('Experiment', 'model_name')

    log_dir = args.path
    logger = Logger(log_dir)

    dataset = Dataset(model_name, **config['Dataset'])

    # evaluator
    evaluator = Evaluator(**config['Evaluator'])

    import model

    MODEL_CLASS = getattr(model, model_name)
    # build model
    model = MODEL_CLASS(dataset, config['Model'], device)

    # test
    model.eval()
    model.restore(logger.log_dir)
    model.logger = logger
    evaluator.evaluate(model, dataset, 'submit')

    logger.info("Saved answer")


if __name__ == '__main__':
    ## For submission
    if os.path.exists('/home/agc2021/dataset/problemsheet_5_00.json'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, default='saves_final/', metavar='P')
        args = parser.parse_args()
        main_submit(args)
    else:
        main_train_test(argv=sys.argv[1:])