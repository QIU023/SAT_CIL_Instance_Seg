tar -czvf - wp_ver_data/ | split -b 10G - wp_ver_data_directory/wp_ver_data.tar.gz

tar -I pigz -cvf wp_ver_data.tar.gz wp_ver_data/