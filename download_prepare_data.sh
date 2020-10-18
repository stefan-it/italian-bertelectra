echo "Downloading preprocessing scripts"
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py

mkdir -p data

cd data

# UD_Italian-ISDT
git clone https://github.com/UniversalDependencies/UD_Italian-ISDT.git
cd UD_Italian-ISDT
git checkout f20fa2b9d45464d7cea9f4191c74ab71f5e68df8

echo "Preprocessing UD_Italian-ISDT (Phase I)"

# Important: replace tabs with spaces to be compatible with run_ner.py script from transformers!
cat it_isdt-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_isdt-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_isdt-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

for model in dbmdz/bert-base-italian-cased dbmdz/bert-base-italian-xxl-cased dbmdz/bert-base-italian-uncased dbmdz/bert-base-italian-xxl-uncased bert-base-multilingual-cased bert-base-multilingual-uncased xlm-roberta-base
do
    mkdir -p $model-data # append data, because preprocess would look into this folder for tokenizer config!!!!

    echo "Preprocessing UD_Italian-ISDT for $model (Phase II)"

    python3 ../../preprocess.py train.txt.tmp $model 128 > $model-data/train.txt
    python3 ../../preprocess.py dev.txt.tmp $model 128 > $model-data/dev.txt
    python3 ../../preprocess.py test.txt.tmp $model 128 > $model-data/test.txt
done

echo "Creating labels for UD_Italian-ISDT (Phase III)"
cat train.txt.tmp dev.txt.tmp test.txt.tmp | grep -v "^$" | cut -d " " -f 2 | sort | uniq > labels.txt

cd .. # from UD_Italian-ISDT

cd .. # from data
