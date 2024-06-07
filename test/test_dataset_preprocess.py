import csv


dataset_characteristics_path = '../datasets/faces_and_identifiers.csv'
test_dataset_identifiers_path = '../datasets/identity_CelebA.txt'
csv_dict = {}
with open(test_dataset_identifiers_path, 'r') as identifiers:
    for identifier in identifiers:
        file_name, current_identifier = identifier.rstrip().split()
        if current_identifier in csv_dict:
            csv_dict[current_identifier].append(file_name)
        else:
            csv_dict[current_identifier] = [file_name]
with open(dataset_characteristics_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['identifier', 'pictures'])
    writer.writeheader()
    for key in csv_dict.keys():
        writer.writerow({'identifier': key, 'pictures': csv_dict[key]})
