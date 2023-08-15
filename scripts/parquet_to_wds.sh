#!/bin/bash

flist=`aws s3 ls s3://s-laion/flickr-narrative-boxes/output/ | awk '{print $4}'`

for i in $flist
do
{
    aws s3 cp s3://s-laion/flickr-narrative-boxes/output/"$i" /fsx/home-shivr/parquet_files/
    tar cvf /fsx/home-shivr/parquet_files/"${i%-*}".tar  /fsx/home-shivr/parquet_files/"$i"
    aws s3 cp /fsx/home-shivr/parquet_files/"${i%-*}".tar s3://laion-west/flickr-narrative-boxes/output_wds/

    rm /fsx/home-shivr/parquet_files/"$i"
    rm /fsx/home-shivr/parquet_files/"${i%-*}".tar
}
done