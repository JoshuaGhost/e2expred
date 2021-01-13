from latent_rationale.config_utils import Config


class BlockConfig:
    def __init__(self, conf):
        self.bert_dir = conf['bert_dir']
        self.warmup_steps = conf['warmup_steps']
        self.use_half_precision = conf['use_half_precision']
        self.max_length = conf['max_length']
        self.cls_head = conf['cls_head']


class MTLConfig(BlockConfig):
    def __init__(self, conf):
        super(MTLConfig, self).__init__(conf)


class CLSConfig(BlockConfig):
    def __init__(self, conf):
        super(CLSConfig, self).__init__(conf)


class SelectorConfig:
    def __init__(self, conf):
        self.selector_type = conf["selector_type"]
        self.dropout = conf['dropout']
        self.dependent_z = conf['dependent-z']
        self.dist = conf['dist']
        self.exp_threshold = conf['exp_threshold']


class E2ExPredConfig(Config):
    def __init__(self, conf):
        print(conf)
        self.data_dir = conf['data_dir']
        self.bert_vocab = conf['bert_vocab']
        self.rebalance_approach = conf['rebalance_approach']
        # self.dataset_name = conf['dataset_name']
        self.sentence_sampling_method = conf["sentence_sampling_method"]
        self.merge_evidence = conf["merge_evidences"]
        self.classes = conf["classes"]

        self.weights_scheduler = conf["weights_scheduler"]
        # self.num_iterations = conf['num_iterations']
        # self.patience = conf['patience']
        # self.batch_size = conf["batch_size"]
        # self.eval_batch_size = conf["eval_batch_size"]
        # self.max_grad_norm = conf["max_grad_norm"]
        # self.cooldown = conf['cooldown']
        # self.weight_decay = conf['weight_decay']

        # self.threshold = conf['threshold']

        # self.lr = conf['lr']
        # self.min_lr = conf['min_lr']
        # self.lr_decay = conf['lr_decay']

        self.lambda_init = conf['lambda_init']
        self.lambda_min = conf['lambda_min']
        self.lambda_max = conf['lambda_max']

        self.soft_selection = conf['soft_selection']

        self.weights = conf['weights']

        self.mtl_conf = MTLConfig(conf['mtl'])
        self.selector_conf = SelectorConfig(conf['selector'])
        self.cls_conf = CLSConfig(conf['cls'])
