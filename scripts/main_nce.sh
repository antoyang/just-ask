#!/bin/bash
#SBATCH --job-name=pt_nce         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --partition=gpu_p2                  # queue
#SBATCH --qos=qos_gpu-t4                # qos: quality of service
#SBATCH --gres=gpu:8         # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=24          # nombre de coeurs à réserver (un quart du noeud)
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=48:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/logs/ptnce_%j.out     # nom du fichier de sortie
#SBATCH --error=/gpfsdswork/projects/rech/msk/urt22vb/JustAsk/logs/ptnce_%j.err      # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
min_words=10
qmax_words=20
min_time=10
max_feats=20
checkpoint_dir="ptnce"
batch_size=128
batch_size_val=3500
n_pair=16
freq_display=20
lr=0.00005
lr_decay=0.9


/gpfswork/rech/msk/urt22vb/miniconda3/bin/python /gpfswork/rech/msk/urt22vb/JustAsk/main_nce.py --dataset="howto100m" --num_thread_reader=16 --epochs=10 --min_words=$min_words --qmax_words=$qmax_words --min_time=$min_time --max_feats=$max_feats --checkpoint_dir=$checkpoint_dir --batch_size=$batch_size --batch_size_val=$batch_size_val --n_pair=$n_pair --freq_display=$freq_display --lr=$lr --lr_decay=$lr_decay