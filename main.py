import sys
import load_pdf
import model
from os.path import join as ospathjoin


def main():
    PDF_NAME = ospathjoin("data_set", "information_retireval.pdf")
    QUERY = sys.argv[1] if len(sys.argv) > 1 else ""

    load_pdf.embed_pdf(PDF_NAME)
    pages_and_chunks, embedding = load_pdf.load_embed()
    tokenizer, lln_model = model.prepare_model()

    print(model.ask(QUERY, pages_and_chunks, embedding, tokenizer, lln_model))


if __name__ == "__main__":
    main()
