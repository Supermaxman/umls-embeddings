import tensorflow as tf

# control options
tf.flags.DEFINE_string("data_dir", "data", "training data directory")
tf.flags.DEFINE_string("model_dir", "model", "model directory")
tf.flags.DEFINE_string("model", "transd", "Model: [transe, transd, distmult]")
tf.flags.DEFINE_string("mode", "disc", "Mode: [disc, gen, gan]")
tf.flags.DEFINE_string("run_name", None, "Run name")
tf.flags.DEFINE_string("summaries_dir", "data/summary", "model summary dir")
tf.flags.DEFINE_integer("batch_size", 1024, "batch size")
tf.flags.DEFINE_integer("val_batch_size", 1024, "val batch size")
tf.flags.DEFINE_bool("load", False, "Load model?")
tf.flags.DEFINE_bool("load_embeddings", False, "Load embeddings?")
tf.flags.DEFINE_string("embedding_file", "data/embeddings.npz", "Embedding matrix npz")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs?")
tf.flags.DEFINE_float("val_proportion", 0.1, "Proportion of training data to hold out for validation")
tf.flags.DEFINE_integer("progress_update_interval", 1000, "Number of batches between progress updates")
tf.flags.DEFINE_integer("val_progress_update_interval", 680, "Number of batches between progress updates")
tf.flags.DEFINE_integer("save_interval", 1800, "Number of seconds between model saves")
tf.flags.DEFINE_integer("batches_per_epoch", 1, "Number of batches per training epoch")
tf.flags.DEFINE_integer("max_batches_per_epoch", None, "Maximum number of batches per training epoch")
tf.flags.DEFINE_string("embedding_device", "gpu", "Device to do embedding lookups on [gpu, cpu]")
tf.flags.DEFINE_string("optimizer", "adam", "Optimizer [adam, sgd]")
tf.flags.DEFINE_string("save_strategy", "epoch", "Save every epoch or saved every"
                                                 " flags.save_interval seconds [epoch, timed]")

# eval control options
tf.flags.DEFINE_string("eval_mode", "save", "Evaluation mode: [save, calc]")
tf.flags.DEFINE_string("eval_dir", "eval", "directory for evaluation outputs")
tf.flags.DEFINE_integer("shard", 1, "Shard number for distributed eval.")
tf.flags.DEFINE_integer("num_shards", 1, "Total number of shards for distributed eval.")
tf.flags.DEFINE_bool("save_ranks", True, "Save ranks? (turn off while debugging)")

# gan control options
tf.flags.DEFINE_string("dis_run_name", None, "Run name for the discriminator model")
tf.flags.DEFINE_string("gen_run_name", "dm-gen", "Run name for the generator model")
tf.flags.DEFINE_string("sn_gen_run_name", "dm-sn-gen", "Run name for the semnet generator model")

# model params
# TODO consider alternatives, like higher gamma
tf.flags.DEFINE_float("learning_rate", 1e-5, "Starting learning rate")
tf.flags.DEFINE_float("decay_rate", 0.96, "LR decay rate")
tf.flags.DEFINE_float("momentum", 0.9, "Momentum")
tf.flags.DEFINE_float("gamma", 0.1, "Margin parameter for loss")
tf.flags.DEFINE_integer("vocab_size", 1726933, "Number of unique concepts+relations")
tf.flags.DEFINE_integer("embedding_size", 50, "embedding size")
tf.flags.DEFINE_integer("energy_norm_ord", 1,
                        "Order of the normalization function used to quantify difference between h+r and t")
tf.flags.DEFINE_integer("max_concepts_per_type", 1000, "Maximum number of concepts to average for semtype loss")
tf.flags.DEFINE_integer("num_generator_samples", 100, "Number of negative samples for each generator example")
tf.flags.DEFINE_string("p_init", "zeros",
                       "Projection vectors initializer: [zeros, xavier, uniform]. Uniform is in [-.1,.1]")

# semnet params
tf.flags.DEFINE_bool("no_semantic_network", False, "Do not add semantic network loss to the graph?")
tf.flags.DEFINE_float("semnet_alignment_param", 0.5, "Parameter to control semantic network loss")
tf.flags.DEFINE_float("semnet_energy_param", 0.5, "Parameter to control semantic network loss")
tf.flags.DEFINE_bool("sn_eval", False, "Train this model with subset of sn to evaluate the SN embeddings?")

# distmult params
tf.flags.DEFINE_float("regularization_parameter", 1e-2, "Regularization term weight")
tf.flags.DEFINE_string("energy_activation", 'sigmoid',
                       "Energy activation function [None, tanh, relu, sigmoid]")


tf.flags.DEFINE_bool("ace_model", False, "Train Atom Concept Embedding model where embeddings are a learned function"
                                        "of the atoms and relation tokens.")

tf.flags.DEFINE_string("encoder_checkpoint",
                       None, #'/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt'
                       "Checkpoint path for embedding encoder network. Start with bert pre-trained.")


tf.flags.DEFINE_string("bert_config", '/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_config.json',
                       "Config path for pre-trained bert model.")
tf.flags.DEFINE_string("bert_vocab", '/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_vocab.txt',
                       "Vocab path for pre-trained bert model.")

tf.flags.DEFINE_integer("encoder_rnn_layers", 1, "Number of layers of encoder rnn.")
tf.flags.DEFINE_integer("encoder_rnn_size", 64, "Size of encoder rnn.")

tf.flags.DEFINE_bool("train_bert", True, "Jointly train bert encoder.")
tf.flags.DEFINE_integer("seed", 1337, "Random seed.")

tf.flags.DEFINE_bool("gpu_memory_growth", False, "Allow gpu memory growth.")
tf.flags.DEFINE_integer("nrof_queued_batches", 10, "Number of batches to queue up in another thread.")


flags = tf.flags.FLAGS
