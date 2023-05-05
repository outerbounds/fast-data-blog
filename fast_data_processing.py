from metaflow import Parameter, FlowSpec, step, S3, resources, batch, conda, conda_base
from table_loader import load_table
from time import time

DEFAULT_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=32000, cpu=8, use_tmpfs=True, tmpfs_size=16000)

@conda_base(libraries={"pyarrow": "11.0.0"}, python="3.10.10")
class FastDataProcessing(FlowSpec):

    num_files = Parameter(
        "num-files", default=1000000, help="Maximum number of files to download"
    )
    url = Parameter("s3src", default=DEFAULT_URL, help="S3 prefix to Parquet files")
    col_name = Parameter("col", default="f_0", help="Column name in Ubiquant data")
    only_download = Parameter("only-download", default=False, is_flag=True)

    @step
    def start(self):
        self.next(self.pandas_, self.polars_, self.duckdb_)

    @conda(libraries={"pandas": "2.0.1"})
    @batch(**RESOURCES)
    @step
    def pandas_(self):
        tic = time()
        table = load_table(
            self.url, self.num_files, num_threads=RESOURCES['cpu'], only_download=self.only_download
        )
        df = table.to_pandas()
        toc = time()
        self.time_elapsed = toc - tic
        self.next(self.join)

    @conda(libraries={"polars": "0.17.11"})
    @batch(**RESOURCES)
    @step
    def polars_(self):
        tic = time()
        import polars as pl
        table = load_table(
            self.url, self.num_files, num_threads=RESOURCES['cpu'], only_download=self.only_download
        )
        df = pl.from_arrow(table)
        toc = time()
        self.time_elapsed = toc - tic
        self.next(self.join)

    @conda(libraries={"python-duckdb": "0.7.1"})
    @batch(**RESOURCES)
    @step
    def duckdb_(self):
        QUERY = """SELECT {} FROM arrow_table;""".format(self.col_name)
        tic = time()
        import duckdb
        arrow_table = load_table(
            self.url, self.num_files, num_threads=RESOURCES['cpu'], only_download=self.only_download
        )
        relation = duckdb.arrow(arrow_table)
        df = relation.query('arrow_table', QUERY).to_df()
        toc = time()
        self.time_elapsed = toc - tic
        self.next(self.join)

    @step
    def join(self, inputs):
        for task in inputs:
            print(f"{task} time elapsed: {task.time_elapsed}")
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    FastDataProcessing()