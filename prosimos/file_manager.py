
class FileManager:
    # class is used only for saving the log file in csv
    def __init__(self, chunk_size, file_writter, additional_column_names = []):
        self.chunk_size = chunk_size
        self.data_buffer = list()
        self.file_writter = file_writter

        self._add_header_row(additional_column_names)

    def add_csv_row(self, csv_row):
        if self.file_writter:
            self.data_buffer.append(csv_row)
            if len(self.data_buffer) >= self.chunk_size:
                self.force_write()

    def _add_header_row(self, additional_column_names = []):
        # additional_column_names is present only in case 
        # there is a provided setup for additional case attributes

        if self.file_writter:
            header_row = ['case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource']
            header_row.extend(additional_column_names)
            self.file_writter.writerow(header_row)

    def force_write(self):
        if self.file_writter:
            self.file_writter.writerows(self.data_buffer)
            self.data_buffer = list()

