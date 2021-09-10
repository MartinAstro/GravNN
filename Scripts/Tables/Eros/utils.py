def remove_rule_lines(table):
    lines = table.splitlines(True)
    new_table = []
    for line in lines:
        if 'rule' in line:
            new_line = '\\hline\n'
        else:
            # Add lines between N samples
            if '625' in line or '1250' in line or '2500' in line:
                new_table.append('\\hline\n')
            new_line = line
            new_line = new_line.replace('pm', '$\\pm$')
            new_line = new_line.replace('textcolor', '\\textcolor')
            new_line = new_line.replace('\{', '{')
            new_line = new_line.replace('\}', '}')
            new_line = new_line.replace('\$', '$')
            new_line = new_line.replace('rt', '\\rt')
            new_line = new_line.replace('bt', '\\bt')
            new_line = new_line.replace('gt', '\\gt')
            new_line = new_line.replace('yt', '\\yt')
            new_line = new_line.replace('lt', '\\lt')
            new_line = new_line.replace('mt', '\\mt')
            new_line = new_line.replace(' 00', ' None')

        new_table.append(new_line)
    return "".join(new_table)

def get_color(value):
    if value >= 1000:
        color = r'rt'
    elif value <1000 and value >= 100:
        color = r'mt'
    elif value < 100 and value >= 10:
        color = r'yt'
    else:
        if value < 5:
            color = r'gt'
        else:
            color = r'lt'
    return color