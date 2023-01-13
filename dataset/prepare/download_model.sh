
mkdir -p pretrained
cd pretrained/

echo -e "The pretrained model files will be stored in the 'pretrained' folder\n"
gdown 1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE

unzip VQTrans_pretrained.zip
echo -e "Cleaning\n"
rm VQTrans_pretrained.zip

echo -e "Downloading done!"