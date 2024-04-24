import handwriting.train_handwriting
import sts.train_sts
import voice.train_voice

def main():
    # Data processing and model training for each attribute
    handwriting.train_handwriting
    sts.train_sts
    voice.train_voice

if __name__ == '__main__':
    main()