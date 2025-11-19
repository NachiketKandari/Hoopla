import argparse

from lib.describe_image import describe_image_command

def main():

    parser = argparse.ArgumentParser(description="Describe Image CLI")

    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--query", type=str, help="Query")

    args = parser.parse_args()

    describe_image_command(args.query, args.image)


if __name__ == "__main__":
    main()