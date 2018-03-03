import sys
from app.runner import Runner


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else ""
    print(model_name)
    verbose = sys.argv[2] == "-v" if len(sys.argv) > 2 else False
    test = sys.argv[2] == "-t" if len(sys.argv) > 2 else False
    runner = Runner(verbose=verbose, model_name=model_name)
    if not test:
        runner.train()
    else:
        runner.test()
