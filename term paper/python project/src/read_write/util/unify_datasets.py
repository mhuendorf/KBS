import csv
import os


def reformat(filename, copyname, fieldnames):
    """Reformat a svg file to order columns differently

    Keyword arguments:
        filename -- the path to the file
        copyname -- the path where the copy should land
        fieldnames -- a list of the new order of fieldnames
    """
    if not os.path.exists(copyname):
        with open(filename, 'r') as infile, open(file_to, 'a') as outfile:
            # output dict needs a list for new column ordering
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            # reorder the header first
            writer.writeheader()
            for row in csv.DictReader(infile):
                # writes the reordered rows to the new file
                writer.writerow(row)


if __name__ == '__main__':

    filename = '../../../res/dota/dota2Train.csv'
    file_to = '../../../res/dota/dota2TrainReordered.csv'
    fieldnames = ['cluster ID', 'game mode', 'game type', 'hero1', 'hero2', 'hero3', 'hero4', 'hero5', 'hero6',
                  'hero7', 'hero8', 'hero9', 'hero10', 'hero11', 'hero12', 'hero13', 'hero14', 'hero15', 'hero16',
                  'hero17', 'hero18', 'hero19', 'hero20', 'hero21', 'hero22', 'hero23', 'hero24', 'hero25',
                  'hero26', 'hero27', 'hero28', 'hero29', 'hero30', 'hero31', 'hero32', 'hero33', 'hero34',
                  'hero35', 'hero36', 'hero37', 'hero38', 'hero39', 'hero40', 'hero41', 'hero42', 'hero43',
                  'hero44', 'hero45', 'hero46', 'hero47', 'hero48', 'hero49', 'hero50', 'hero51', 'hero52',
                  'hero53', 'hero54', 'hero55', 'hero56', 'hero57', 'hero58', 'hero59', 'hero60', 'hero61',
                  'hero62', 'hero63', 'hero64', 'hero65', 'hero66', 'hero67', 'hero68', 'hero69', 'hero70',
                  'hero71', 'hero72', 'hero73', 'hero74', 'hero75', 'hero76', 'hero77', 'hero78', 'hero79',
                  'hero80', 'hero81', 'hero82', 'hero83', 'hero84', 'hero85', 'hero86', 'hero87', 'hero88',
                  'hero89', 'hero90', 'hero91', 'hero92', 'hero93', 'hero94', 'hero95', 'hero96', 'hero97',
                  'hero98', 'hero99', 'hero100', 'hero101', 'hero102', 'hero103', 'hero104', 'hero105', 'hero106',
                  'hero107', 'hero108', 'hero109', 'hero110', 'hero111', 'hero112', 'hero113', 'team won']
    reformat(filename, file_to, fieldnames)

    filename = '../../../res/dota/dota2Test.csv'
    file_to = '../../../res/dota/dota2TestReordered.csv'
    reformat(filename, file_to, fieldnames)

    filename = '../../../res/mushroom/agaricus-lepiota.data'
    file_to = '../../../res/mushroom/agaricus-lepiota-Reordered.data'
    fieldnames = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing',
                  'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                  'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                  'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'class']
    reformat(filename, file_to, fieldnames)
