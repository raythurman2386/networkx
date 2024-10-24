import os
from sys import argv
import arcpy
from Logger3 import Logger


def manage_attribute_rules(workspace):
    arcpy.env.overwriteOutput = True
    source_gdb = os.path.join(workspace, "template.gdb")
    target_gdb = os.path.join(workspace, "Working/ingest_300269.gdb")
    rules_dir = os.path.join(workspace, "attribute_rules")
    if not os.path.exists(rules_dir):
        os.makedirs(rules_dir)

    logger = Logger(workspace, "UpdateAttributeRules")

    try:
        if not arcpy.Exists(source_gdb) or not arcpy.Exists(target_gdb):
            raise ValueError("One or both geodatabases do not exist")

        # Walk through source geodatabase
        for dirpath, dirnames, filenames in arcpy.da.Walk(source_gdb,
                                                          datatype=["Table", "FeatureClass"]):
            for filename in filenames:
                try:
                    source = os.path.join(dirpath, filename)
                    target = os.path.join(target_gdb, filename)

                    if not arcpy.Exists(target):
                        logger.msg(f"Target {filename} does not exist in target geodatabase. Skipping...")
                        continue

                    logger.msg(f"Processing {filename}")
                    desc = arcpy.da.Describe(source)

                    # Check if the source has attribute rules
                    if len(desc['attributeRules']) > 0:
                        csv_path = os.path.join(rules_dir, f"{filename}_rules.csv")
                        try:
                            logger.msg(f"\tExporting rules for {filename} to CSV")
                            arcpy.ExportAttributeRules_management(
                                in_table=source,
                                out_csv_file=csv_path
                            )

                            # Delete existing rules in target
                            existing_rules = arcpy.Describe(target).attributeRules
                            for rule in existing_rules:
                                try:
                                    arcpy.DeleteAttributeRule_management(target, rule['name'])
                                except:
                                    pass

                            logger.msg(f"\tImporting rules from CSV to {filename}")
                            arcpy.ImportAttributeRules_management(
                                target_table=target,
                                csv_file=csv_path
                            )

                            logger.msg(f"\tSuccessfully processed rules for {filename}")

                        except Exception as e:
                            logger.msg(f"\tError processing rules for {filename}: {str(e)}")
                            continue
                    else:
                        logger.msg(f"\tNo attribute rules found in {filename}")

                except Exception as e:
                    logger.msg(f"Error processing {filename}: {str(e)}")
                    continue

        logger.msg("Script completed successfully")

    except Exception as e:
        logger.msg(f"Major error in processing: {str(e)}")
        raise


if __name__ == "__main__":
    manage_attribute_rules(*argv[1:])