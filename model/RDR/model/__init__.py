import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M    #从当前模块的model子模块中导入DDPM类，并重命名为M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m    #返回创建的模型实例m

def create_model_val(opt):
    from model.RDA.model.model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
