import typer
import json
import os
from pathlib import Path
import typer
import re
import shutil
import pandas as pd

app = typer.Typer()

@app.command('strip')
def strip_json(directory_path: Path = typer.Argument(..., help="Path to the directory to scan")):
    def strip_image(file):
        try:
            with open(file, "r") as read_file:
                data = json.load(read_file)
            if data.get('imageData'):
                data['imageData'] = None
                # Serializing json
                json_object = json.dumps(data, indent=2)
                # Writing to sample.json
                with open(file, "w") as outfile:
                    outfile.write(json_object)
                return True
            return True
        except:
            return True
    try:
        # Check if the directory exists
        if not directory_path.exists():
            typer.echo("Directory does not exist.")
            return

        # List all files in the directory
        typer.echo(f"Files in {directory_path}:")
        for file in directory_path.rglob('**/*.json'):
            if file.stat().st_size>2000:
                if strip_image(file):
                    typer.echo(file)
    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}")

@app.command('file')
def file_json(directory_path: Path = typer.Argument(..., help="moves json files into years"),
               destination: Path = typer.Argument(..., help="moves json files into years")):
    try:
        # Check if the directory exists
        if not directory_path.exists():
            typer.echo("Directory does not exist.")
            return

        # List all files in the directory
        typer.echo(f"Files in {directory_path}:")
        regex =re.compile(r'(\d{8})T\d+')
        for file in directory_path.rglob('*.json'):
            if not file.is_dir():
                match = regex.search(file.name)
                if match:
                    extracted_date = match.group(1)
                    destinationdir = destination  / extracted_date 
                    destinationdir.mkdir(parents=True,exist_ok=True)
                    dest =destinationdir / file.name
                    if not dest.exists():
                        shutil.copy(file,dest)
                else:
                    print("Date pattern not found in the file name.")
    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}")

@app.command('reverse')
def reverse_json(source_path: Path = typer.Argument(..., help="moves json files into years"),
               dest_path: Path = typer.Argument(..., help="moves json files into years")):
    try:
        # Check if the directory exists
        if (not source_path.exists()) | (not dest_path.exists()):
            typer.echo("Directory does not exist.")
            return

        # List all files in the directory
        typer.echo(f"Files in {source_path}:")
        regex =re.compile(r'(\d{8})T\d+')
        source = list(source_path.rglob('*.json'))
        input_files = pd.DataFrame(source,columns=['Path'])
        input_files['Name']=input_files.Path.apply(lambda x:x.name)
        input_files['TimeStamp']=input_files['Name'].str.extract(r'(?P<TimeStamp>\d+T\d+)')
        input_files['Date'] = input_files.TimeStamp.str.extract(r'(?P<date>\d+)')
        input_files =input_files.set_index('TimeStamp').sort_index()
        dest = list(dest_path.rglob('*.json'))
        dest_files = pd.DataFrame(dest,columns=['Path'])
        dest_files['Name']=dest_files.Path.apply(lambda x:x.name)
        dest_files['TimeStamp']=dest_files['Name'].str.extract(r'(?P<TimeStamp>\d+T\d+)')
        dest_files =dest_files.set_index('TimeStamp').sort_index()
        new_files = input_files.loc[input_files.index.difference(dest_files.index)]
        for index,row in new_files.iterrows():
            dest_dir = dest_path / row.Date
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / row.Name
            shutil.copy(row.Path,dest)

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}")        

if __name__ == "__main__":
    app()

