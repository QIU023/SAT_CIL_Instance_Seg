import os, sys

def upload_hdfs_cmd(source_file, target_file):
    cmd = f'hdfs dfs -copyFromLocal {source_file} hdfs://haruna/home/byte_Data_Video/qiuyiqiao/{target_file}'
    os.system(cmd)

def download_hdfs_cmd(source_file, target_file):
    cmd = f'hdfs dfs -copyToLocal hdfs://haruna/home/byte_Data_Video/qiuyiqiao/{source_file} {target_file}'
    os.system(cmd)

def remove_hdfs_cmd(target_file):
    cmd = f'hdfs dfs -rm hdfs://haruna/home/byte_Data_Video/qiuyiqiao/{target_file}'
    os.system(cmd)

def ls_hdfs_cmd(target_dir):
    cmd = f'hdfs dfs -ls hdfs://haruna/home/byte_Data_Video/qiuyiqiao/{target_dir}'
    os.system(cmd)

def mkdir_hdfs_cmd(target_dir):
    cmd = f'hdfs dfs -mkdir hdfs://haruna/home/byte_Data_Video/qiuyiqiao/{target_dir}'
    os.system(cmd)

process_id = int(sys.argv[1])
number_tot_thread = int(sys.argv[2])
execute_type = str(sys.argv[3])
# process_gap = int(1000 / number_tot_thread)
# process_begin = process_gap*process_id
# process_end = process_gap*(process_id+1)

if __name__ == '__main__':
    postfix_a = process_id // 26
    postfix_b = process_id % 26
    postfix = chr(ord('a') + postfix_a) + chr(ord('a') + postfix_b)
    fname = 'wp_ver_shape_error_data.tar.gz'+postfix
    if execute_type == 'remove':
        data_path = 'wp_ver_data_directory/' + fname
        remove_hdfs_cmd(data_path)
    elif execute_type == 'upload':
        data_dir = '../wp_ver_data_directory/' + fname
        upload_hdfs_cmd(data_dir, fname)
    elif execute_type == 'download':
        data_dir = '../wp_ver_data_directory/' + fname
        download_hdfs_cmd(fname, data_dir)
    elif execute_type == 'list':
        ls_hdfs_cmd('wp_ver_data_directory')
    else:
        raise NotImplementedError
