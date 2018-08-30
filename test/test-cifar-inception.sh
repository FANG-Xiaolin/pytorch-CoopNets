cd ..
python main.py -test -img_size 32 -test_size 60000 -ckpt_gen ./test/ckpt_gen_cifar.pth -ckpt_des ./test/ckpt_des_cifar.pth -output_dir ./test-fid -langevin_step_num_des 8
cd test
echo 'Computing Inception Score...'
python inception_model.py


