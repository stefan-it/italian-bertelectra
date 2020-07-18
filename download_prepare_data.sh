export CASED_MODEL=dbmdz/bert-base-italian-xxl-cased
export UNCASED_MODEL=dbmdz/bert-base-italian-xxl-uncased

mkdir -p data

cd data

echo "Downloading preprocessing scripts"
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess_wikiann.py
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess_wikiner.py
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/create_random_split.py

# WikiANN Rahimi et. al splits
mkdir -p wikiann
cd wikiann
wget https://schweter.eu/storage/wikiann/it.tar.gz
tar -xzf it.tar.gz

echo "Preprocessing WikiAnn (Phase I)"
python3 ../preprocess_wikiann.py train > train.txt.tmp
python3 ../preprocess_wikiann.py dev > dev.txt.tmp
python3 ../preprocess_wikiann.py test > test.txt.tmp

echo "Preprocessing WikiAnn (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt

python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from wikiann

# WikiNER WP3
mkdir -p wikiner
cd wikiner
wget https://github.com/dice-group/FOX/raw/master/input/Wikiner/aij-wikiner-it-wp3.bz2
bunzip2 aij-wikiner-it-wp3.bz2

echo "Preprocessing WikiNER (Phase I)"
python3 ../preprocess_wikiner.py aij-wikiner-it-wp3 > all_wikiner 

echo "Preprocessing WikiNER (Phase II)"
python3 ../create_random_split.py all_wikiner 5

mkdir -p random_split_{1,2,3,4,5}

mv 1_* random_split_1
mv 2_* random_split_2
mv 3_* random_split_3
mv 4_* random_split_4
mv 5_* random_split_5

echo "Preprocessing WikiNER (Phase III)"
python3 ../preprocess.py random_split_1/1_train.txt $CASED_MODEL 128 > random_split_1/train-cased.txt
python3 ../preprocess.py random_split_1/1_train.txt $UNCASED_MODEL 128 > random_split_1/train-uncased.txt

python3 ../preprocess.py random_split_2/2_train.txt $CASED_MODEL 128 > random_split_2/train-cased.txt
python3 ../preprocess.py random_split_2/2_train.txt $UNCASED_MODEL 128 > random_split_2/train-uncased.txt

python3 ../preprocess.py random_split_3/3_train.txt $CASED_MODEL 128 > random_split_3/train-cased.txt
python3 ../preprocess.py random_split_3/3_train.txt $UNCASED_MODEL 128 > random_split_3/train-uncased.txt

python3 ../preprocess.py random_split_4/4_train.txt $CASED_MODEL 128 > random_split_4/train-cased.txt
python3 ../preprocess.py random_split_4/4_train.txt $UNCASED_MODEL 128 > random_split_4/train-uncased.txt

python3 ../preprocess.py random_split_5/5_train.txt $CASED_MODEL 128 > random_split_5/train-cased.txt
python3 ../preprocess.py random_split_5/5_train.txt $UNCASED_MODEL 128 > random_split_5/train-uncased.txt

cd .. # from wikiner

# UD_Italian-ParTUT
git clone https://github.com/UniversalDependencies/UD_Italian-ParTUT.git
cd UD_Italian-ParTUT 
git checkout 38500550ac066d54cef76885e737ed69cbcec835

echo "Preprocessing UD_Italian-ParTUT (Phase I)"
cat it_partut-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_partut-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_partut-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

echo "Preprocessing UD_Italian-ParTUT (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt

python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt

python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from UD_Italian-ParTUT

# UD_Italian-TWITTIRO
git clone https://github.com/UniversalDependencies/UD_Italian-TWITTIRO.git
cd UD_Italian-TWITTIRO
git checkout 4ece137f5ce2bbf585d0b7311c614fdc5ec0047c

echo "Preprocessing UD_Italian-TWITTIRO (Phase I)"
cat it_twittiro-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_twittiro-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_twittiro-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

echo "Preprocessing UD_Italian-TWITTIRO (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt

python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt

python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from UD_Italian-TWITTIRO

# UD_Italian-PoSTWITA
git clone https://github.com/UniversalDependencies/UD_Italian-PoSTWITA.git
cd UD_Italian-PoSTWITA
git checkout 662b23564b7e0d359b3b73889d31b0c1063a2e03

echo "Preprocessing UD_Italian-PoSTWITA (Phase I)"
cat it_postwita-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_postwita-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_postwita-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

echo "Preprocessing UD_Italian-PoSTWITA (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt

python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt

python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from UD_Italian-PoSTWITA

# UD_Italian-VIT
git clone https://github.com/UniversalDependencies/UD_Italian-VIT.git
cd UD_Italian-VIT
git checkout f84efef31ca0757a5b16ed1625770d6771ddff2c

echo "Preprocessing UD_Italian-VIT (Phase I)"
cat it_vit-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_vit-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_vit-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

echo "Preprocessing UD_Italian-VIT (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt

python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt

python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from UD_Italian-VIT

# UD_Italian-ISDT
git clone https://github.com/UniversalDependencies/UD_Italian-ISDT.git
cd UD_Italian-ISDT
git checkout f20fa2b9d45464d7cea9f4191c74ab71f5e68df8

echo "Preprocessing UD_Italian-ISDT (Phase I)"
cat it_isdt-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > train.txt.tmp
cat it_isdt-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > dev.txt.tmp
cat it_isdt-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" > test.txt.tmp

echo "Preprocessing UD_Italian-ISDT (Phase II)"
python3 ../preprocess.py train.txt.tmp $CASED_MODEL 128 > train-cased.txt
python3 ../preprocess.py train.txt.tmp $UNCASED_MODEL 128 > train-uncased.txt

python3 ../preprocess.py dev.txt.tmp $CASED_MODEL 128 > dev-cased.txt
python3 ../preprocess.py dev.txt.tmp $UNCASED_MODEL 128 > dev-uncased.txt

python3 ../preprocess.py test.txt.tmp $CASED_MODEL 128 > test-cased.txt
python3 ../preprocess.py test.txt.tmp $UNCASED_MODEL 128 > test-uncased.txt

cd .. # from UD_Italian-ISDT

cd .. # from data
