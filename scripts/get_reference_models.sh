# bn flow model was trained with cuDNN v4, we will use the weights upgraded to cuDNN v5
wget -O models/bn_inception_anet_2016_temporal.caffemodel.v5 https://yjxiong.blob.core.windows.net/anet-2016/bn_inception_anet_2016_temporal.caffemodel.v5 --no-check-certificate
wget -O models/bn_inception_anet_2016_temporal_deploy.prototxt https://yjxiong.blob.core.windows.net/anet-2016/bn_inception_anet_2016_temporal_deploy.prototxt --no-check-certificate
wget -O models/resnet200_anet_2016.caffemodel https://yjxiong.blob.core.windows.net/anet-2016/resnet200_anet_2016.caffemodel --no-check-certificate
wget -O models/resnet200_anet_2016_deploy.prototxt https://yjxiong.blob.core.windows.net/anet-2016/resnet200_anet_2016_deploy.prototxt --no-check-certificate