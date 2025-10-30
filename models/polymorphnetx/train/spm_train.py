
import argparse, sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_prefix", default="configs/spm")
    ap.add_argument("--vocab_size", type=int, default=50000)
    args = ap.parse_args()

    user_symbols = ["[AGENT]","[MEM]","[PLUGIN]","[VERIFY]","[POLICY]","[IMG]","[/IMG]","[AUD]","[/AUD]"]
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        user_defined_symbols=user_symbols,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    print(f"Trained {args.model_prefix}.model")

if __name__ == "__main__":
    main()
