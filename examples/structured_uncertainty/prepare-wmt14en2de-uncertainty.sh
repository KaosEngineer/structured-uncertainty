#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
    "http://www.statmt.org/wmt14/medical-task/khresmoi-summary-test-set.tgz"
    "http://www.statmt.org/wmt14/medical-task/khresmoi-query-test-set.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
    "khresmoi-summary-test-set.tgz"
    "khresmoi-query-test-set.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ "$1" == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
    OUTDIR=wmt14_en_de
else
    OUTDIR=wmt17_en_de
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

#echo "pre-processing train data..."
#for l in $src $tgt; do
#    rm $tmp/train.tags.$lang.tok.$l
#    for f in "${CORPORA[@]}"; do
#        cat $orig/$f.$l | \
#            perl $NORM_PUNC $l | \
#            perl $REM_NON_PRINT_CHAR | \
#            perl $TOKENIZER -threads 24 -a -l $l >> $tmp/train.tags.$lang.tok.$l
#    done
#done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/test.$l
    head -n 1500 $tmp/test.$l > $tmp/test-h1.$l
    tail -n +1501 $tmp/test.$l > $tmp/test-h2.$l
    echo ""
    grep '<seg id' $orig/khresmoi-summary-test-set/khresmoi-summary-dev.${l}.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/bio-ks-dev.$l
    echo ""
    grep '<seg id' $orig/khresmoi-summary-test-set/khresmoi-summary-test.${l}.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/bio-ks-test.$l
    echo ""
    cat $tmp/bio-ks-dev.$l $tmp/bio-ks-test.$l > $tmp/bio-ks.$l
    echo ""
    cat $orig/librispeech/test-clean.txt | \
        sed -e  's/[0-9]*\-[0-9]*\-[0-9]* //g' | \
        sed 's/.*/\L&/' | \
        sed -e 's/^\(.\)/\U\1/g' | \
        sed -e "s/$/\./" | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/librispeech-tc.$l
    echo ""
    cat $orig/librispeech/test-other.txt | \
        sed -e  's/[0-9]*\-[0-9]*\-[0-9]* //g' | \
        sed 's/.*/\L&/' | \
        sed -e 's/^\(.\)/\U\1/g' | \
        sed -e "s/$/\./" | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/librispeech-tp.$l
done
echo ""
    grep '<seg id' $orig/test-full/newstest2014-fren-ref.fr.sgm | \
    sed -e 's/<seg id="[0-9]*">\s*//g' | \
    sed -e 's/\s*<\/seg>\s*//g' | \
    sed -e "s/\’/\'/g" | \
perl $TOKENIZER -threads 24 -a -l fr > $tmp/test.fr
echo ""



#echo "splitting train and valid..."
#for l in $src $tgt; do
#    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
#    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
#done

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
#rm -f $TRAIN
#for l in $src $tgt; do
#    cat $tmp/train.$l >> $TRAIN
#done
#
#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    #for f in train.$L valid.$L test.$L test-h1.$L test-h2.$L bio-ks-dev.$L bio-ks-test.$L bio-ks.$L librispeech-tc.$L librispeech-tp.$L ; do
    for f in test.$L test-h1.$L test-h2.$L bio-ks-dev.$L bio-ks-test.$L bio-ks.$L librispeech-tc.$L librispeech-tp.$L ; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.fr > $tmp/bpe.test.fr

#perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
#perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
    cp $tmp/bpe.test-h1.$L $prep/test-h1.$L
    cp $tmp/bpe.test-h2.$L $prep/test-h2.$L
    cp $tmp/bpe.bio-ks-dev.$L $prep/bio-ks-dev.$L
    cp $tmp/bpe.bio-ks-test.$L $prep/bio-ks-test.$L
    cp $tmp/bpe.bio-ks.$L $prep/bio-ks.$L
    cp $tmp/librispeech-tc.$L $prep/librispeech-tc.$L
    cp $tmp/librispeech-tp.$L $prep/librispeech-tp.$L

    cat $prep/test.$L | python permute_sentence.py > $prep/test-perm.$L
done
cp $tmp/bpe.test.fr $prep/test.fr


cd $prep

#Make language-switched forms of the data
cp test.de test-deen.en
cp test.en test-deen.de
cp test.en test-enen.de
cp test.en test-enen.en
cp test.de test-dede.en
cp test.de test-dede.de
cp test.de test-ende.en
cp test.en test-ende.de

#Make BPE-permuted forms of the data.
cp test-perm.de test-perm-ende.de
cp test-perm.en test-perm-ende.en
cp test.en test-perm-de.en
cp test-perm.de test-perm-de.de
cp test-perm.en test-perm-en.en
cp test.de test-perm-en.de
cd ../