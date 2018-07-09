#######################################################################
#
# An example of creating Excel Bar charts with Python and XlsxWriter.
#
# Copyright 2013-2018, John McNamara, jmcnamara@cpan.org
#
import xlsxwriter

workbook = xlsxwriter.Workbook('chart_bar.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': 1})

# Add the worksheet data that the charts will refer to.
headings = ['Number', 'Batch 1', 'Batch 2']
data = [
    [2, 3, 4, 5, 6, 7],
    [10, 40, 50, 20, 10, 50],
    [30, 60, 70, 50, 40, 30],
]

worksheet.write_row('A1', headings, bold)
worksheet.write_column('A2', data[0])
worksheet.write_column('B2', data[1])
worksheet.write_column('C2', data[2])

#######################################################################
#
# Create a new bar chart.
#
chart1 = workbook.add_chart({'type': 'bar'})

# Configure the first series.
chart1.add_series({
    'name':       '=Sheet1!$B$1',
    'categories': '=Sheet1!$A$2:$A$7',
    'values':     '=Sheet1!$B$2:$B$7',
})

# Configure a second series. Note use of alternative syntax to define ranges.
chart1.add_series({
    'name':       ['Sheet1', 0, 2],
    'categories': ['Sheet1', 1, 0, 6, 0],
    'values':     ['Sheet1', 1, 2, 6, 2],
})

# Add a chart title and some axis labels.
chart1.set_title({'name': 'Results of sample analysis'})
chart1.set_x_axis({'name': 'Test number'})
chart1.set_y_axis({'name': 'Sample length (mm)'})

# Set an Excel chart style.
chart1.set_style(11)

# Insert the chart into the worksheet (with an offset).
worksheet.insert_chart('D2', chart1, {'x_offset': 25, 'y_offset': 10})

#######################################################################
#
# Create a stacked chart sub-type.
#
chart2 = workbook.add_chart({'type': 'bar', 'subtype': 'stacked'})

# Configure the first series.
chart2.add_series({
    'name':       '=Sheet1!$B$1',
    'categories': '=Sheet1!$A$2:$A$7',
    'values':     '=Sheet1!$B$2:$B$7',
})

# Configure second series.
chart2.add_series({
    'name':       '=Sheet1!$C$1',
    'categories': '=Sheet1!$A$2:$A$7',
    'values':     '=Sheet1!$C$2:$C$7',
})

# Add a chart title and some axis labels.
chart2.set_title ({'name': 'Stacked Chart'})
chart2.set_x_axis({'name': 'Test number'})
chart2.set_y_axis({'name': 'Sample length (mm)'})

# Set an Excel chart style.
chart2.set_style(12)

# Insert the chart into the worksheet (with an offset).
worksheet.insert_chart('D18', chart2, {'x_offset': 25, 'y_offset': 10})

#######################################################################
#
# Create a percentage stacked chart sub-type.
#
chart3 = workbook.add_chart({'type': 'bar', 'subtype': 'percent_stacked'})

# Configure the first series.
chart3.add_series({
    'name':       '=Sheet1!$B$1',
    'categories': '=Sheet1!$A$2:$A$7',
    'values':     '=Sheet1!$B$2:$B$7',
})

# Configure second series.
chart3.add_series({
    'name':       '=Sheet1!$C$1',
    'categories': '=Sheet1!$A$2:$A$7',
    'values':     '=Sheet1!$C$2:$C$7',
})

# Add a chart title and some axis labels.
chart3.set_title ({'name': 'Percent Stacked Chart'})
chart3.set_x_axis({'name': 'Test number'})
chart3.set_y_axis({'name': 'Sample length (mm)'})

# Set an Excel chart style.
chart3.set_style(13)

# Insert the chart into the worksheet (with an offset).
worksheet.insert_chart('D34', chart3, {'x_offset': 25, 'y_offset': 10})

workbook.close()
