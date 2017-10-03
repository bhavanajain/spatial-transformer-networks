import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description="Train Spatial Transformer Networks using different architectures")

	parser.add_argument('--stn_arch', '-stn', dest='STN_ARCH', 
							action='store', default='CNN', choices=['CNN', 'FCN', 'None'])

	parser.add_argument('--classifier_arch', '-clsfr', dest='CLASSIFIER_ARCH',
							action='store', default='CNN', choices=['CNN', 'FCN']) 

	parser.add_argument('--n_epochs', '-ne', dest='N_EPOCHS', 
							action='store', type=int, default=100)

	parser.add_argument('--learning_rate', '-lr', dest='LEARNING_RATE',
							action='store', type=float, default=0.001)

	parser.add_argument('--reg', '-r', dest='REG',
							action='store', default='L2', choices=['None', 'L1', 'L2'])

	parser.add_argument('--reg_param', '-beta', dest='BETA',
							action='store', type=float, default=1e-3)

	parser.add_argument('--use_pretrained', '-pre', dest='PRETRAINED',
							action='store', type=bool, default=False)

	parser.add_argument('--model_path', '-model', dest='MODEL_PATH',
							action='store')

	parser.add_argument('--data_dir', '-data', dest='DATA_DIR',
							action='store', default='/home/bhvjain/stn_models/data/')

	parser.add_argument('--samples_dir', '-samples', dest='SAMPLES_DIR',
							action='store', default='./samples')

	parser.add_argument('--logfile', '-lf', dest='LOGFILE',
							action='store', default='./logs.txt')

	parser.add_argument('--model_save', dest='MODEL_SAVE',
							action='store', default='all', choices=['best', 'all'])
                                            
        def restricted_float(x):
            x = float(x)
            if x < 0.0 or x > 1.0:
                raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % x)

	parser.add_argument('--gpu_fraction', dest='GPU_FRAC',
							action='store', type=restricted_float, default=1)

	args = parser.parse_args()
	return args

ARGS = parse_arguments()




