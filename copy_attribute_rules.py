import os
from sys import argv

import arcpy
from Logger3 import Logger


def copy_attribute_rules(workspace):
    arcpy.env.overwriteOutput = True
    # TODO: Determine how we plan on doing our attribute validations.
    # This is currently set up to use a template gdb that houses all of our validations and will
    # pull them to whichever target database is needed if the feature layers exist.
    # This does work but DOES NOT transfer table to table rules that use SQL and Check Parameters
    source_gdb = os.path.join(workspace, "template.gdb")
    target_gdb = os.path.join(workspace, "Working/ingest_300269.gdb")
    logger = Logger(workspace, "UpdateAttributeRules")

    try:
        if not arcpy.Exists(source_gdb) or not arcpy.Exists(target_gdb):
            raise ValueError("One or both geodatabases do not exist")

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
                        for ar in desc['attributeRules']:
                            try:
                                logger.msg(f"\tProcessing rule: {ar.name}")
                                # Get all necessary properties from the rule
                                rule_properties = {
                                    "name": ar.name,
                                    # Esri added EsriArt to the type field for some reason: EsriArtValidation
                                    "type": ar.type[7:].upper(),
                                    "script_expression": ar.scriptExpression,
                                    "error_number": ar.errorNumber,
                                    "error_message": ar.errorMessage,
                                    "severity": ar.severity,
                                    "triggering_events": ar.triggeringEvents,
                                    "description": ar.description,
                                    "tags": ar.tags
                                }

                                # Remove any existing rule with the same name
                                try:
                                    arcpy.DeleteAttributeRule_management(target, ar.name)
                                except:
                                    pass

                                arcpy.AddAttributeRule_management(
                                    in_table=target,
                                    **rule_properties
                                )

                                logger.msg(f"\tSuccessfully added rule: {ar.name}")

                            except Exception as e:
                                logger.msg(f"\tError processing rule {ar.name}: {str(e)}")
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
    copy_attribute_rules(*argv[1:])