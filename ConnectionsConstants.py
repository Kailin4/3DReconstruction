'''
This file contains the constants that are used for indexing
into the connections data structure.

The dictionary is used for neat debugging the fields of the
data structure.
'''


CONNECTION_ATTRIBUTES = 6
VIEW_ID_1 = 0
VIEW_ID_2 = 1
RELATIVE_ROTATION = 2
RELATIVE_TRANSLATION = 3
SOURCE_POINTS = 4
DESTINATION_POINTS = 5



connectionAttributesDict = {
	'VIEW_ID_1': VIEW_ID_1,
	'VIEW_ID_2': VIEW_ID_2,
	'RELATIVE_ROTATION': RELATIVE_ROTATION,
	'RELATIVE_TRANSLATION': RELATIVE_TRANSLATION,
	'SOURCE_POINTS': SOURCE_POINTS,
	'DESTINATION_POINTS': DESTINATION_POINTS
}

E = [1e-6]