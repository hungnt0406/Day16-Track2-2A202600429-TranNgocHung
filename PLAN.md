
  # Plan hoàn thành Lab 16 AWS theo fallback CPU + LightGBM                                                                                           
                                                                                                                                                      
  ## Tóm tắt                                                                                                                                          
                                                                                                                                                      
  - Đi theo Phần 7 của README_aws.md:162, nhưng tối ưu cho mục tiêu nộp bài nhanh: giữ nguyên phần lớn hạ tầng, chỉ đổi GPU node thành CPU node, rồi SSH vào máy private để chạy benchmark LightGBM.
  - Repo hiện tại chưa có benchmark.py, chưa có terraform/lab-key.pub, và Bastion trong terraform/main.tf:152 dùng Ubuntu AMI nên SSH user đúng là ubuntu, không phải ec2-user.
  - Mặc định không cố làm ALB phục vụ inference ở fallback CPU. ALB có thể unhealthy; điều đó không chặn deliverables của phần 7.                     
                                                                                                                                                      
  ## Chuẩn bị local                                                                                                                                   
                                                                                                                                                      
  1. Hoàn tất phần IAM của README: tạo IAM group/user có quyền EC2, VPC, ELB, IAM; lấy access key.                                                    
  2. Cài aws CLI và terraform, chạy aws configure với region us-east-1, rồi xác nhận bằng aws sts get-caller-identity.                                
  3. Tạo SSH key ở ngoài repo để không lộ private key khi nộp bài:                                                                                    
                                                                                                                                                      
  ssh-keygen -t rsa -b 4096 -f ~/.ssh/ai-lab-key -N ""                                                                                                
  cp ~/.ssh/ai-lab-key.pub terraform/lab-key.pub                                                                                                      
  chmod 400 ~/.ssh/ai-lab-key                                                                                                                         
                                                                                                                                                      
  4. Tạo Kaggle account, vào Settings -> API -> Create New Token, giữ sẵn username và key để nhập trên EC2.                                           
                                                                                                                                                      
  ## Chỉnh repo cho fallback CPU                                                                                                                      
                                                                                                                                                      
  - Sửa terraform/main.tf:175:                                                                                                                        
      - Đổi aws_instance.gpu_node.instance_type từ g4dn.xlarge sang r5.2xlarge.                                                                       
      - Thay AMI của gpu_node từ Deep Learning AMI sang Amazon Linux 2023 bằng một data source mới aws_ami.al2023 với owner amazon và filter theo al2023-ami-2023*-x86_64.
      - Giữ nguyên tên resource/output hiện tại; gpu_private_ip vẫn dùng làm IP của CPU node để tránh sửa lan rộng.                                   
      - Bỏ templatefile("${path.module}/user_data.sh", ...) khỏi gpu_node, hoặc thay bằng bootstrap rỗng tối thiểu. Mục tiêu là không chạy script vLLM/NVIDIA cũ trên máy CPU.
  - Không sửa ALB, target group, security group, outputs, variables.tf, outputs.tf, hay user_data.sh. Đó là đường ít chỉnh nhất để apply nhanh.       
  - Tạo local benchmark.py trong repo. Script này phải cố định hành vi:                                                                               
      - Đọc creditcard.csv bằng pandas và đo load_data_time.                                                                                          
      - Tách X/y, split train_test_split(test_size=0.2, stratify=y, random_state=42).                                                                 
      - Train lightgbm.LGBMClassifier với objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, early stopping 50 rounds trên validation set.
      - Tính best_iteration, auc_roc, accuracy, f1, precision, recall.                                                                                
      - Đo inference_latency_1_row_ms bằng trung bình 100 lần predict_proba trên 1 sample sau 10 warm-up calls.                                       
      - Đo inference_throughput_1000_rows_per_sec bằng 20 lần predict_proba trên 1000 samples sau 5 warm-up calls.                                    
      - In toàn bộ metrics ra stdout và ghi cùng dữ liệu vào benchmark_result.json.                                                                   
                                                                                                                                                      
  ## Triển khai và chạy benchmark                                                                                                                     
                                                                                                                                                      
  1. Vào terraform/, chạy terraform init, terraform validate, rồi terraform apply.                                                                    
  2. Ghi thời điểm bấm yes; đây là mốc để sau này ghi tổng thời gian triển khai theo đường CPU.                                                       
  3. Sau khi apply xong, lấy bastion_public_ip và gpu_private_ip.                                                                                     
  4. SSH thẳng từ local vào private CPU node bằng ProxyJump, không copy private key lên bastion:                                                      
                                                                                                                                                      
  ssh -i ~/.ssh/ai-lab-key -J ubuntu@<bastion_public_ip> ec2-user@<gpu_private_ip>                                                                    
                                                                                                                                                      
  5. Trên CPU node, cài môi trường:                                                                                                                   
                                                                                                                                                      
  sudo dnf update -y                                                                                                                                  
  sudo dnf install -y python3 python3-pip                                                                                                             
  pip3 install --upgrade pip                                                                                                                          
  pip3 install lightgbm scikit-learn pandas numpy kaggle                                                                                              
  mkdir -p ~/ml-benchmark                                                                                                                             
                                                                                                                                                      
  6. Tạo ~/.kaggle/kaggle.json, chmod 600, rồi tải dataset:                                                                                           
                                                                                                                                                      
  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/                                                                      
                                                                                                                                                      
  7. Từ local, upload benchmark.py vào ~/ml-benchmark/ bằng scp có ProxyJump, rồi chạy:                                                               
                                                                                                                                                      
  python3 ~/ml-benchmark/benchmark.py                                                                                                                 
                                                                                                                                                      
  8. Chụp màn hình terminal ngay khi script in xong metrics. Sau đó copy benchmark_result.json từ EC2 về local.                                       
  9. Không debug ALB hoặc /v1/chat/completions; fallback CPU không yêu cầu endpoint LLM hoạt động.                                                    
                                                                                                                                                      
  ## Deliverables và kiểm tra hoàn tất                                                                                                                
                                                                                                                                                      
  - terraform validate pass trước khi apply.                                                                                                          
  - terraform apply hoàn tất mà không dính quota GPU.                                                                                                 
  - SSH vào được Bastion bằng user ubuntu, vào CPU node bằng user ec2-user.                                                                           
  - ~/ml-benchmark/creditcard.csv tồn tại và python3 benchmark.py thoát mã 0.                                                                         
  - benchmark.py in đủ các metric của README và sinh benchmark_result.json.                                                                           
  - Thu đủ 5 deliverables:                                                                                                                            
      - Screenshot terminal chạy python3 benchmark.py.                                                                                                
      - File benchmark_result.json.                                                                                                                   
      - Screenshot AWS Billing sau đúng 1 giờ kể từ lúc terraform apply hoàn tất.                                                                     
      - Source terraform/ đã chỉnh cho CPU, kèm local benchmark.py.                                                                                   
      - Báo cáo ngắn 5–10 dòng nêu lý do dùng CPU thay GPU và tóm tắt metric.                                                                         
  - Khi đã lấy đủ artifacts, chạy terraform destroy ngay. Không nộp terraform.tfstate, token, hay private key ~/.ssh/ai-lab-key.                      
                                                                                                                                                      
  ## Assumptions                                                                                                                                      
                                                                                                                                                      
  - Bạn đang bắt đầu từ repo hiện tại và mới chỉ có AWS account, chưa có IAM user/CLI/SSH key usable cho lab.                                         
  - Mục tiêu là nộp bài nhanh, không refactor fallback CPU thành một service sau ALB.                                                                 
  - Region cố định là us-east-1.                                                                                                                      