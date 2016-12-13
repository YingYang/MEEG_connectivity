#!/bin/tcsh

#foreach i (`seq 1 1 7`)
foreach i (`seq 13 1 18`)
	python Script_apply_on_Scene_MEEG.py Subj$i MEG
end

#python Script_apply_on_Scene_MEEG.py Subj2 MEG
#python Script_apply_on_Scene_MEEG.py Subj3 MEG

