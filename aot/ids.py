"""Print body instances (box in 960x540 review coords) for train frames: python aot/ids.py 2 5 7"""
import json, sys
for i in sys.argv[1:]:
    m = json.load(open(f"work/teacher_train/meta/aot_train_{int(i):04d}.json"))
    print(f"#{int(i)}: " + " ".join(f"{e['id']}{'g' if e['kind']=='gear' else ''}[{int(e['box'][0]/2)},{int(e['box'][1]/2)},{int(e['box'][2]/2)},{int(e['box'][3]/2)}]a{e['area']//1000}k" for e in m if e['kind'] == 'body'))
