import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multi Modal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify Image Embedding")
    verify_parser.add_argument("image", type=str, help="Relative path of the image")
    
    image_search_parser = subparsers.add_parser("image_search", help="Search Using Image")
    image_search_parser.add_argument("image", type=str, help="Relative path of the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            image_search_command(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()