import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data_2/data/data/")
    parser.add_argument("--save_dir", type=str, default="./t5_des/")
    parser.add_argument("--schema_path", type=str, default="../data_2/data/data/schema.json")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default='t5-small')
    parser.add_argument("--num_beams", type=int, default=5)

    args = parser.parse_args()
    return args
