We provide a simple bash script that will `untar` only LibriQuote necessary files in each `small`, `medium` and `large` LibriLight archives.

```bash
# Go to LibriLight path where each .tar is stored
cd $PATH_TO_LIBRILIGHT

# Launch the bash script
bash $PATH_TO_THIS_REPOSITORY/librilight_matching/match.sh \
$PATH_TO_THIS_REPOSITORY/librilight_matching/matched_librilight_name.txt
```