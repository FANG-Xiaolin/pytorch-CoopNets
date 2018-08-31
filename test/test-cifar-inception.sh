cd ..
python main.py -test -score -img_size 32 -test_size 60000 -ckpt_gen ./test/ckpt_gen_cifar.pth -ckpt_des ./test/ckpt_des_cifar.pth -output_dir ./test-inception -langevin_step_num_des 8
echo 'Computing Inception Score...'
python -m test.inception_model


