import glob


def concatenate_tables(tables, prefix):
    tables.sort()
    new_table = ""
    for i in range(len(tables)):
        table = tables[i]
        with open(table, "r") as f:
            lines = f.readlines()
        if i == 0:
            new_table += "".join(lines[:9])  # Keep the opener
            new_table += "\t&\t&\t&Exterior\t&\\\\\n"
            new_table += "".join(lines[9:-3])  # Keep the opener
        elif i == len(tables) - 1:
            new_table += "\\hline\n"
            new_table += "\t&\t&\t&Surface\t&\\\\\n"
            new_table += "".join(lines[9:])  # Keep the closer
        else:
            new_table += "\\hline\n"
            new_table += "\t&\t&\t&Interior\t&\\\\\n"
            new_table += "".join(lines[9:-3])

    file_name = "Notes/PINN_Asteroid_Journal/Assets/meta_" + prefix + "nn_table.tex"
    with open(file_name, "w") as f:
        f.write("".join(new_table))
    print(new_table)


def main():
    prefix = "sh_"
    tables = glob.glob("Notes/PINN_Asteroid_Journal/Assets/**" + prefix + "table**.tex")
    concatenate_tables(tables, prefix)

    prefix = "pinn_"
    tables = glob.glob("Notes/PINN_Asteroid_Journal/Assets/" + prefix + "**.tex")
    concatenate_tables(tables, prefix)

    prefix = "transformer_"
    tables = glob.glob("Notes/PINN_Asteroid_Journal/Assets/" + prefix + "**.tex")
    concatenate_tables(tables, prefix)

    prefix = "annealing_transformer_"
    tables = glob.glob("Notes/PINN_Asteroid_Journal/Assets/" + prefix + "**.tex")
    concatenate_tables(tables, prefix)

    prefix = "pinn40_"
    tables = glob.glob("Notes/PINN_Asteroid_Journal/Assets/" + prefix + "**.tex")
    concatenate_tables(tables, prefix)


if __name__ == "__main__":
    main()
