import sys, csv

if __name__ == "__main__":
  with open(sys.argv[1], 'rb') as csvfile, open(sys.argv[2], "w") as outfile:
    reader = csv.reader(csvfile, delimiter='\t')
    writer = csv.writer(outfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    total = 0
    has_correction = 0
    for row in reader:
      if len(row) >= 5:   # some of the data rows are bad
        if len(row) >= 6:   # if it has a correction
          for correction in row[5:]:  # write a row for each one
            total += 1
            if row[4].strip() != correction.strip():
              has_correction += 1
            writer.writerow((row[4].strip(), correction.strip()))
        else: # if it does not have a correction
          total += 1
          writer.writerow((row[4].strip(), row[4].strip()))
    print "%s total pairs, %s has corrections" % (total, has_correction)

  # This is a sanity check for the new data file. The two print statements
  # should be the same.
  with open(sys.argv[2], "r") as new_data:
    reader = csv.reader(new_data,
                        delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    total = 0
    has_correction = 0
    for row in reader:
      total += 1
      if row[0] != row[1]:
        has_correction += 1
    print "%s total sentences, %s has corrections" % (total, has_correction)
