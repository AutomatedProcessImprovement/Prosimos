
class FileManager:
    def __init__(self, chunk_size, file_writter):
        self.chunk_size = chunk_size
        self.data_buffer = list()
        self.file_writter = file_writter

    def add_csv_row(self, csv_row):
        if self.file_writter:
            self.data_buffer.append(csv_row)
            if len(self.data_buffer) >= self.chunk_size:
                self.force_write()

    def force_write(self):
        if self.file_writter:
            self.file_writter.writerows(self.data_buffer)
            self.data_buffer = list()

