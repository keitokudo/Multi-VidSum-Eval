import argparse

def main(args):
    while True:
        try:
            line = input()
            if line.endswith(".") or line.endswith("?") or line.endswith("!"):
                print(line)
            else:
                print(f"{line}.")
                
        except EOFError:
            break
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
