rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading extractors"
gdown --fuzzy https://drive.google.com/file/d/1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug/view
gdown --fuzzy https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view


unzip t2m.zip
unzip kit.zip

echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip
echo -e "Downloading done!"