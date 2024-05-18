##########
#G14
data_dir=$(pwd)/G14_model  # change here data_dir if required
mkdir -p "$data_dir"  # make dir if it does not exists
echo "Downloading data in ${data_dir}"
printf "============"
echo "model G14 checkpoints"
#wget "https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt"
modg14="https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt"
wget -O "${data_dir}/model.pt" "$modg14"  # download

################
#L14
data_dir=$(pwd)/L14_model  # change here data_dir if required
mkdir -p "$data_dir"  # make dir if it does not exists
echo "Downloading data in ${data_dir}"
printf "============"
echo "model G14 checkpoints"
#wget "https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt"
modl14="https://huggingface.co/OpenShape/openshape-pointbert-vitl14-rgb/resolve/main/model.pt"
wget -O "${data_dir}/model.pt" "$modl14"  # download


################
#B32
data_dir=$(pwd)/B32_model  # change here data_dir if required
mkdir -p "$data_dir"  # make dir if it does not exists
echo "Downloading data in ${data_dir}"
printf "============"
echo "model G14 checkpoints"
#wget "https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt"
modb32="https://huggingface.co/OpenShape/openshape-pointbert-vitb32-rgb/resolve/main/model.pt"
wget -O "${data_dir}/model.pt" "$modb32"  # download




echo "Finished"
