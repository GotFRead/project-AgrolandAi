import app.Application_create_test_datasets as app
import os 

print('Начало тестирования')
def check_back_to_root_dirs():
    old_path = os.getcwd()
    app.back_to_root_dirs()
    # app.back_to_root_dirs()
    if old_path.split('\\')[:2] == os.getcwd().split('\\')[:2]:
        print(f'Successfully, {os.getcwd()} and {old_path}')
        return True
    else:
        print('Failury')
        return False

check_back_to_root_dirs()