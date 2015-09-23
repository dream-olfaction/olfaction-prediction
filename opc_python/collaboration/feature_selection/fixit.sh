# Leander's files all_features_*.csv contain the NSPDK features for the wrong CIDs.
# This is an attempt to fix those files... it doesn't work yet.
for SET in training validation; do
  cat all_features_${SET}.csv | cut -f 1-7307,9745- -d , > all_features_${SET}_nonspdk.csv.tmp
  cut -f 1 -d , all_features_${SET}.csv >cid_${SET}.csv.tmp
done
# Make a file with first the CID, then the NSPDK Gramian
#   The header
echo "CID,nspdk$(seq -s ",nspdk" 1 2437)" > nspdk_gram_cid.csv.tmp
#   The data
zcat ../../../data/derived/nspdk_r3_d4_unaug_gramian.mtx.gz \
  | paste -d " " ../../../data/derived/nspdk_cid.csv - \
  | head -n 476 | tr " " , >> nspdk_gram_cid.csv.tmp

python3 fixit.py
# The following is easier in Python...
# tail -n +2 all_features_training_nonspdk.csv.tmp | sort -t , > all_features_training_nonspdk_noheader.csv.tmp
# join --check-order -j 1 -t , all_features_training_nonspdk_noheader.csv.tmp nspdk_gram_cid.csv.tmp > all_features_training_fixed.csv
# for SET in training validation; do
#   join --check-order --header -j 1 -t , all_features_${SET}_nonspdk.csv.tmp nspdk_gram_cid.csv.tmp > all_features_${SET}_fixed.csv
# done
