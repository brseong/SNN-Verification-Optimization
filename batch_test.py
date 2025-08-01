from typing import Literal, LiteralString
from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from adv_rob_fmnist_module import run_test as run_test_fmnist
from numpy import arange
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from time import strftime, localtime

required_arguments:list[LiteralString] = "test_type prefix".split()

TestType = Literal["iris", "mnist", "fmnist"]
def run_test(cfg:CFG, test_type:TestType="mnist"):
    match test_type:
        case "iris": return run_test_iris(cfg)
        case "mnist": return run_test_mnist(cfg)
        case "fmnist": return run_test_fmnist(cfg)
        case _: raise NotImplementedError(f"Test type must be in {TestType}.")

def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta-max", dest="delta_max", type=int, default=1)
    parser.add_argument("--repeat", dest="repeat", type=int, default=1)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=14)
    parser.add_argument("--test-type", dest="test_type", type=str)
    parser.add_argument("--z3", dest="z3", action="store_true", default=False)
    parser.add_argument("--milp", dest="milp", action="store_true", default=False)

    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = [strftime('%m%d%H%M', localtime())]
    if getattr(parser, "repeat") > 1: words.append(f"rep_{parser.repeat}")
    if hasattr(parser, "test_type"): words.append(parser.test_type)
    if hasattr(parser, "prefix"): words.append(parser.prefix)
    
    prefix: str
    if parser.z3: prefix = "z3"
    elif parser.milp: prefix = "milp"
    else: prefix = "np"
    words.append(prefix)
    return '_'.join(words)

if __name__ == "__main__":
    parser = parse()
    
    if getattr(parser, "repeat") < 1:
        raise ValueError("repeat must be greater than 0.")
    
    if all(hasattr(parser, s) for s in required_arguments):
        for iteration in range(parser.repeat):
            run_test(CFG(log_name=prepare_log_name(parser),
                        seed=parser.seed,
                        num_samples=parser.num_samples,
                        deltas=(parser.delta_max,),
                        z3=parser.z3,
                        milp=parser.milp),
                    test_type=parser.test_type)
    else:
        raise ValueError("Not appropriate arguments.")
