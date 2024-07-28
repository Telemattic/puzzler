import csv
import mysql.connector
from tqdm import tqdm

# CREATE TABLE `tabs` (
# `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
# `label` VARCHAR(64) NOT NULL, 
# `tab_no` INT NOT NULL, 
# `col_no` INT, 
# `row_no` INT);
# CREATE TABLE `facts` (
# `dst_id` INT NOT NULL,
# `src_id` INT NOT NULL,
# `mse` NUMERIC(8,3),
#  FOREIGN KEY (`dst_id`) REFERENCES `tabs`(`id`),
#  FOREIGN KEY (`src_id`) REFERENCES `tabs`(`id`)
#  );

class Uploader:

    def __init__(self, cnx):
        self.cnx = cnx
        self.tabs = dict()
        self.facts = []
        self.pending_commit = 0

    def upload_csv(self, path):

        n = 0
        with open(path) as f:
            for _ in f:
                n += 1
            n -= 1
        print(f"{n} rows")

        cursor = self.cnx.cursor()
        cursor.execute('TRUNCATE `facts`')
        cnx.commit()
            
        cursor = self.cnx.cursor()
        cursor.execute('DELETE FROM `tabs`')
        cnx.commit()

        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=n, ascii=True):
                dst_id = self.get_or_insert_tab(row['dst_label'], int(row['dst_tab_no']), int(row['dst_col_no']), int(row['dst_row_no']))
                src_id = self.get_or_insert_tab(row['src_label'], int(row['src_tab_no']), int(row['src_col_no']), int(row['src_row_no']))
                mse = row['mse']
                if mse == '':
                    mse = None
                else:
                    mse = min(9999,float(mse))
                self.insert_fact(dst_id, src_id, mse)
                
        self.flush_inserts()

    def insert_fact(self, dst_id, src_id, mse):
        self.facts.append((dst_id, src_id, mse))
        if len(self.facts) >= 1000:
            self.flush_inserts()

    def flush_inserts(self):

        if self.pending_commit:
            self.cnx.commit()
            self.pending_commit = 0
        
        if len(self.facts) == 0:
            return

        sql = "INSERT INTO `facts` (`dst_id`, `src_id`, `mse`) VALUES (%s, %s, %s)"
        
        cursor = self.cnx.cursor()
        cursor.executemany(sql, self.facts)
        cnx.commit()

        self.facts = []

    def get_or_insert_tab(self, label, tab_no, col_no, row_no):
        k = (label, tab_no)
        v = self.tabs.get(k)
        if v is not None:
            return v

        cursor = self.cnx.cursor()

        sql = "INSERT INTO `tabs` (`label`, `tab_no`, `col_no`, `row_no`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (label, tab_no, col_no, row_no))

        self.pending_commit += 1
        # self.cnx.commit()

        v = cursor.lastrowid
        self.tabs[k] = v
        return v

print("There should be an SSH tunnel to MySQL on pizzabox.lan:\n\tssh -L 3306:127.0.0.1:3306 eldridge@pizzabox.lan")
cnx = mysql.connector.connect(user='puzzler', password='puzzler', host='localhost', database='puzzler')

uploader = Uploader(cnx)
uploader.upload_csv('tabs1000.csv')
