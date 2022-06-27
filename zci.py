import spack.environment as env
import spack.mirror
import spack.hash_types as ht
import spack.binary_distribution as binary
import llnl.util.tty as tty
import yaml
import sys
import os
import copy
import json
from itertools import chain
import multiprocessing
from collections import defaultdict


def setup_parser(subparser): 
  subparser.add_argument(
    '--mirror',
    dest='mirror',
    required=False,
    help="""mirror to check""")
  subparser.add_argument(
    '--output', '-o',
    dest='output',
    default="./dag.json",
    required=False,
    help="""output file""")


description = "generate a build manifest containing each spack job needed to install an environment"
section = "build"
level = "short"


def is_rebuild_required(o):
  name, spec, mirror = o
  print("Checking if rebuild is required: {}".format(name))
  needs_rebuild = False
  try:
    tty._msg_enabled = False
    tty._error_enabled = False
    if binary.needs_rebuild(spec, mirror, True):
      needs_rebuild = True
  except Exception as e:
    needs_rebuild = True

  return (name, needs_rebuild)


def zci(parser, args, **kwargs):

  def jobname(s):
    return "{}@{}%{}-{} {}".format(s.name, s.version, s.compiler, s.dag_hash(7), s.architecture)

  def specfilename(s):
    return "{}-{}.spec.json".format(s.name, s.dag_hash(7))

  tty._warn_enabled = False

  e = spack.cmd.require_active_env(cmd_name='zci')
  with spack.concretize.disable_compiler_existence_check():
    with e.write_transaction():
      e.concretize()
      e.write()

  css = [cs for _, cs in e.concretized_specs()]


  m = defaultdict(list)
  rebuilds = {}
  roots = {}
  gpu = {}

  for cs in css:
    roots[jobname(cs)] = True
    if 'cuda' in cs.variants and str(cs.variants['cuda']) == '+cuda':
      gpu[jobname(cs)] = 'cuda'
    elif 'rocm' in cs.variants and str(cs.variants["rocm"]) == '+rocm' :
      gpu[jobname(cs)] = 'rocm'
    
    for s in cs.traverse(deptype=all):
      rjob = jobname(s)
      rebuilds[rjob] = s
      for d in s.dependencies(deptype=all):
        djob = jobname(d)
        rebuilds[djob] = d

  if args.mirror:
    jobs = [(k, v, args.mirror) for k, v in rebuilds.items()]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(is_rebuild_required, jobs)
    pool.close()
    pool.join()

    for (name, needs_rebuild) in results:
      if not needs_rebuild:
        del rebuilds[name]

  tty.msg("Rebuild required for: {}".format(len(list(rebuilds.keys()))))

  needsMap = {}
  staged = {}
  stages = []
  current_stage = 0
  while len(list(rebuilds.keys())) > 0:
    stage = []

    for k, v in rebuilds.items():
      deps = list(v.dependencies(deptype=all))

      cleared = 0
      needs = []
      for d in deps:
        n = jobname(d)
        if n not in rebuilds:
          cleared += 1
        if n in staged:
          needs.append(n)

      outstanding_needs = len(deps) - cleared
      assert outstanding_needs >= 0, "Needs < 0 for {}".format(k)
      if outstanding_needs == 0:
        stage.append(k)
        needsMap[k] = needs

    assert len(stage) > 0
    stages.append(stage)

    len0 = len(rebuilds)
    for n in stage:
      staged[n] = rebuilds[n]
      del rebuilds[n]
    len1 = len(rebuilds)
    assert len0 - len1 == len(stage), "{}, {}, expected decrease = {}".format(len0,len1,len(stage))

  if len(stages) == 0:
    tty.msg("All specs in environment are up-to-date!")
    return 0

  y = {
    "stages": list(range(len(stages))),
    "jobs": {}
  }
  for ii, jobs in enumerate(stages):
    for j in jobs:
      spec = staged[j]

      tags = []
      if j in gpu:
        tags.append(gpu[j])

      y["jobs"][j] = {
        "stage": ii,
        "spec_name": spec.name,
        "version": str(spec.version),
        "is_root": j in roots,
        "spec_file": specfilename(spec),
        "dag_hash": spec.dag_hash(),
        "full_hash": spec.full_hash(),
        "build_hash": spec.build_hash(),
        "needs": needsMap[j],
        "tags": ",".join(tags)
      }

  basename = os.path.basename(args.output)
  dirname = os.path.dirname(args.output)
  if len(dirname) == 0:
    dirname = "."

  os.makedirs(dirname, exist_ok=True)
  output_path = os.path.abspath(os.path.join(dirname, basename))
  
  with open(output_path, "w") as fs:
    fs.write(json.dumps(y, indent=1))
  
  specs_dir_basename = "specs"
  specs_dir = os.path.abspath(os.path.join(dirname, specs_dir_basename))
  os.makedirs(specs_dir, exist_ok=True)

  ss = list(chain.from_iterable(stages))
  for j in ss:
    s = staged[j]
    f = os.path.join(specs_dir, specfilename(s))
    with open(f, 'w') as fs:
      fs.write(s.to_json(hash=ht.build_hash) )

  return 0
