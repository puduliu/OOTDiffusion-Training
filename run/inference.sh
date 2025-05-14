# python run_ootd.py --model_path examples/model/049205_0.jpg --cloth_path examples/garment/00055_00.jpg --model_type dc --category 0 --scale 2.0 --sample 4
python run_ootd.py --model_path examples/model/model_1.png --cloth_path examples/garment/04825_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_ootd.py --model_path examples/model/model_1.png --cloth_path examples/garment/00055_00.jpg --model_type hd --category 0 --scale 2.0 --sample 20

python run_ootd.py --model_path examples/model/model_1.png --cloth_path examples/garment/03032_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_ootd_ip_vton.py --model_path examples/model/model_1.png --cloth_path examples/garment/04825_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_ip_vton.py --model_path examples/model/model_1.png --cloth_path examples/garment/04825_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_vton_768.py --model_path examples/model/model_1.png --cloth_path examples/garment/04825_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_vton.py --model_path examples/model/model_1.png --cloth_path examples/garment/04825_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_vton.py --model_path examples/model/model_1.png --cloth_path examples/garment/03032_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_ip_vton.py --model_path examples/model/model_1.png --cloth_path examples/garment/03032_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4

python run_vton_768.py --model_path examples/model/model_1.png --cloth_path examples/garment/03032_00.jpg --model_type hd --category 0 --scale 2.0 --sample 4


python inference.py --model_type hd --category 0 --scale 2.0 --sample 4 --gpu_id 4

scale指的guidance scale
