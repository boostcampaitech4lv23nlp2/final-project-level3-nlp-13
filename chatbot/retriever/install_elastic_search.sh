# sudo 설치
apt-get update && apt-get -y install sudo

# 패키지 색인을 업데이트
sudo apt update
# HTTPS를 통해 리포지토리에 액세스하는 데 필요한 apt-transport-https 패키지를 설치
sudo apt install apt-transport-https

# OpenJDK 8 설치
sudo apt install openjdk-8-jdk
java -version  # openjdk version "1.8.0_191"

# OpenPGP 암호화 툴 설치
apt-get install -y gnupg2

# Elasticsearch 저장소의 GPG key를 사용해 설치 (GPG key를 암호화/복호화 프로그램이라고 이해하고 넘김)
# 리눅스 배포판에는 기본적으로 GPG가 설치되어 있음
# OK가 반환되어야 함 (맨 뒤 add까지 명령어로 입력해야 함)
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -  

# Elasticsearch 저장소를 시스템에 추가
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list

# elasticsearch 설치
sudo apt update
sudo apt install elasticsearch  

# elasticsearch 시작(설치 완료 후 자동으로 시작되지 않음)
service elasticsearch start

# 경로 이동해서 nori 형태소분석기 설치
cd /usr/share/elasticsearch
bin/elasticsearch-plugin install analysis-nori

# elasticsearch 재시작 (형태소분석기 설치 후 재시작이 필수)
service elasticsearch restart

# curl 명령어 설치
sudo install curl

# Elasticsearch가 실행 중인지 확인
curl "localhost:9200"

# Python Elasticsearch Client 설치
pip install elasticsearch
