import re
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

from ase.calculators.calculator import FileIOCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class FLAPW(FileIOCalculator):
    """
    FLAPW (Full-potential Linearized Augmented Plane Wave) 法のコードを実行するための
    ASE (Atomic Simulation Environment) カスタム電子状態計算クラス。
    
    バージョン履歴:
    0.0.1 : 初期実装
    0.0.2 : waitコードのアップデート
    0.0.3 : ファイル操作の最適化 (copyからmoveへの移行) とコード整理
    """
    implemented_properties = ["energy"]
    name = "flapw"

    def __init__(
        self,
        command=None,       # None の場合は mpi パラメータから自動構築されます
        input_file="lapwin",
        output_file="lapwout",
        jspins=2,
        star_cutoff=9.8,
        pw_cutoff=3.9,
        kpts=(5, 5, 5),
        smearing=0.001,
        xc="gga",
        starting_state="AFM",
        maxiter=100,
        mixing="A",
        representation="SR",
        mpi=False,          # True で mpiexec を使用
        **kwargs,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.jspins = jspins
        self.star_cutoff = star_cutoff
        self.pw_cutoff = pw_cutoff
        self.kpts = kpts
        self.smearing = smearing
        self.xc = xc
        self.starting_state = starting_state
        self.maxiter = maxiter
        self.mixing = mixing
        self.representation = representation
        self.mpi = mpi

        # command が明示的に指定されていない場合、mpi パラメータを基に自動構築
        if command is None:
            if self.mpi:
                command = "module load intel && module load impi && mpiexec ./pflapw"
            else:
                command = "./flapw"

        super().__init__(
            command=command,
            **kwargs,
        )

    # =========================================================================
    # 1. インプットファイル生成関連メソッド
    # =========================================================================

    def write_input(self, atoms, properties=None, system_changes=None):
        """ASEが計算実行前に呼び出すインプット書き出し用のメインルーチン"""
        super().write_input(atoms, properties, system_changes)

        # 計算用ディレクトリ（self.directory）内にインプットファイルを定義
        infile = Path(self.directory) / self.input_file
        self.write_lapwin(atoms, infile)

    def write_lapwin(self, atoms, filename):
        """FLAPW用のメインインプットファイル (lapwin) を作成する"""
        formula = atoms.get_chemical_formula()
        
        # 単位変換係数 (Angstrom から Bohr)
        ANG_TO_BOHR = 1.889726125
        cell_bohr = atoms.cell.array * ANG_TO_BOHR
        lattice_vectors = cell_bohr.copy()

        symbols = atoms.get_chemical_symbols()
        frac_coords = atoms.get_scaled_positions()

        with open(filename, "w") as f:
            # タイトルと計算モード
            f.write(f"Title: {formula}\n")
            f.write("Mode: bulk  auto                           !bulk/film auto/kpts/base\n")

            # 格子ベクトルの書き出し
            f.write("*** lattice vectors **********************\n")
            f.write("1.00000\n")
            for vec in lattice_vectors:
                f.write(f"{vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")

            # 原子座標の設定（内部座標/フラクショナル座標を使用）
            f.write("*** atomic number cartesian or internal coordinates\n")
            f.write("set pos file:F, name:fconst_pos1.dat\n")
            f.write("internal\n")

            # 同一元素が連続する場合は元素記号を省略して位置のみを出力
            prev_symbol = None
            for sym, pos in zip(symbols, frac_coords):
                x, y, z = pos
                if sym != prev_symbol:
                    f.write(f"{sym:<2} {x:12.8f} {y:12.8f} {z:12.8f}\n")
                    prev_symbol = sym
                else:
                    f.write(f"   {x:12.8f} {y:12.8f} {z:12.8f}\n")

            # 空間群と一般オプションの出力
            f.write("*** SPACE GROUP ***************************\n")
            f.write(f"Representation:{self.representation} ,option:default              !SR/ZR/FR/UR SR:default\n")
            f.write("*** GENERAL OPTIONS ***********************\n")
            f.write("Density of states:F, option:default\n")
            f.write("Band structure:F, option:default\n")
            f.write("Density plot:F, option:default\n")
            f.write("Slice analysis:F, option:set\n")
            f.write("Force calculation:F, option:default\n")
            f.write("Geometry optimization:F, option:default\n")
            f.write("Nudged elastic band:F, option:default\n")
            f.write("Force constant calculation:F, option:default\n")
            f.write("Electron-phonon coupling:F, option:set\n")
            f.write("External E field:F, option:set\n")
            f.write("External H field:F, option:set\n")
            f.write("Jellium potential:F, option:set\n")
            f.write("Electric field gradient:F, option:default\n")
            f.write("L matrix calculation:F, option:default\n")
            f.write("P matrix calculation:F, option:default\n")
            f.write("J matrix calculation:F, option:default\n")
            f.write("Second variational +U:F, option:set\n")
            f.write("Second variational SOC:F, option:default\n")
            f.write("Dispersion correction:F, option:default\n")
            f.write("Magnetic dipole-dipole:F, option:default\n")
            f.write("Noncollinear Magnetism:F, option:set\n")
            f.write("Equi-density constraint:F, option:set\n")

            # --- マフィンティン半径 (RMT) テーブルの読み込み ---
            # カレントディレクトリにある "README_MT-default" から RMT と lmax をパース
            rmt_table = {}
            with open("README_MT-default") as h:
                for line in h:
                    m = re.search(
                        r"element\('([A-Za-z ]+)'\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)",
                        line
                    )
                    if m:
                        symbol = m.group(1).strip()
                        lmax = int(m.group(2))
                        rmt = float(m.group(3))
                        rmt_table[symbol] = {"lmax": lmax, "rmt": rmt}

            # 重複を除いた構成元素リストを作成
            species = []
            for s in atoms.get_chemical_symbols():
                if s not in species:
                    species.append(s)

            # 基底関数オプションの書き出し
            f.write("*** BASES *********************************\n")
            f.write(f"star-function cut-off:{self.star_cutoff}\n")
            f.write(f"jspins={self.jspins}\n")
            f.write("e_float:T, xo:T\n")
            f.write("nwin=1\n")
            f.write(f"plane-wave cut off:{self.pw_cutoff}\n")
            f.write("number of states:0\n")
            f.write("lapw parameters:set\n")

            # 各元素に対応する RMT パラメータを出力
            for elem in species:
                rmt = rmt_table[elem]["rmt"]
                lmax = rmt_table[elem]["lmax"]
                f.write(f"{elem:<2} rmt={rmt:.2f} lmax={lmax} 0.\n")
                f.write("                   0.\n")
            
            # K点メッシュとミキシングオプション
            f.write("*** K-POINTS ******************************\n")
            f.write("k-point generator:S, option:default\n")
            f.write(f"Smearing:G, parameter:{self.smearing}\n")
            f.write("Time-reversal symmetry:T\n")
            f.write("Division along internal axis (each window new line)\n")
            f.write(f"  {self.kpts[0]}   {self.kpts[1]}   {self.kpts[2]}\n")

            f.write("*** MIXING  OPTIONS ***********************\n")
            f.write(f"(B)royden or (S)traight mixing for density:{self.mixing}\n")
            f.write(f"Maximum number of iterations:{self.maxiter}  20\n")
            f.write("Mixing parameter:0.\n")
            f.write("Convergency: 0.\n")

            # スピン分極・初期磁気モーメント設定
            f.write("*** OPTIONS FOR SPIN-POLARIZED CASE *******\n")
            f.write("Spin-options:set\n")
            f.write("Initial spin polarization:T\n")
            f.write(f"Starting state and values:{self.starting_state}\n")

            # 主要な遷移金属の初期磁気モーメント定義
            mag_init = {"Fe": 3.0, "Co": 3.0, "Ni": 2.0, "Mn": 4.0, "Cr": 3.0}

            for sym in species:
                moment = mag_init.get(sym, 0.0)
                f.write(f" {sym:<2}  {moment:4.1f}    0.0   0.0    0.0\n")
            
            f.write("Mixing parameter: 0.\n")

            # 計算詳細設定と交換相関（XC）汎関数
            f.write("*** ADVANCED SETTINGS *********************\n")
            f.write("Advanced setup:set\n")
            f.write("Output:redu\n")
            f.write("Check potential and density:F\n")
            f.write(f"Exchange correlation:{self.xc}\n")
            f.write("frcor:F, ctail:F\n")
            f.write("*** END ***********************************\n")

    # =========================================================================
    # 2. 計算結果の読み込み・監視メソッド
    # =========================================================================

    def read_results(self):
        """計算結果出力ファイル (lapwout) から全エネルギーを読み込む"""
        outfile = Path(self.directory) / self.output_file
        text = outfile.read_text()

        # 出力ファイルから最終イテレーションの全エネルギー (単位: Hartree) を正規表現で抽出
        matches = re.findall(
            r"total energy for it=\s*\d+:\s*([-0-9.Ee+]+)\s*htr",
            text,
            flags=re.IGNORECASE,
        )

        if not matches:
            raise RuntimeError("Could not find total energy in lapwout")

        energy_hartree = float(matches[-1])

        # 単位を Hartree から eV (ASE標準) に変換して保持
        HARTREE_TO_EV = 27.211386245988
        self.results["energy"] = energy_hartree * HARTREE_TO_EV

    def wait_for_flapw(self, interval=30):
        """FLAPWのバックグラウンド計算が完了するまで指定間隔で待機・監視する"""
        outfile = Path(self.directory) / self.output_file

        # 計算開始前に古い出力ファイルが存在すれば誤認防止のため削除
        #if outfile.exists():
        #    print("[Python] Found old file! Deleting it now...")
        #    outfile.unlink()

        while True:
            if outfile.exists():
                text = outfile.read_text(errors="ignore")

                # エラー（停止）の検知
                if "Stop:" in text:
                    raise RuntimeError("FLAPW stopped.")

                # 正常終了の検知
                if "FLAPW calculations were done" in text:
                    print("FLAPW finished.")
                    break
                
            print(f"[Python] Checking... File finished? No. Waiting {interval}s...")
            time.sleep(interval)

    # =========================================================================
    # 3. 外部ユーティリティコマンド実行メソッド
    # =========================================================================

    def copy_scf(self):
        """FLcopy コマンドを実行して自己無撞着場(SCF)ファイルを複製する"""
        subprocess.run(
            ["FLcopy", "scf"],
            cwd=self.directory,
            check=True,
        )

        workdir = Path(self.directory)
        scfdir = workdir / "SCF"
        scfdir.mkdir(exist_ok=True)

        for f in workdir.iterdir():
            if f.is_file() and f.name.endswith("scf"):
                shutil.move(f, scfdir / f.name)
        
        return scfdir
    
    def copy_soc(self):
        """FLcopy コマンドを実行してsocファイルを複製する"""
        subprocess.run(
            ["FLcopy", "soc"],
            cwd=self.directory,
            check=True,
        )
        workdir = Path(self.directory)
        socdir = workdir / "SOC"
        socdir.mkdir(exist_ok=True)

        for f in workdir.iterdir():
            if f.is_file() and f.name.endswith("soc"):
                shutil.move(f, socdir / f.name)
        
        return socdir

    def clean(self):
        """FLclean コマンドを実行して計算ディレクトリをクリアする"""
        subprocess.run(
            ["FLclean"],
            cwd=self.directory,
            check=True,
        )

    def restart_scf(self):
        """FLrst コマンドを実行してSCF計算をリスタート(再開)状態にする"""
        subprocess.run(
            ["FLrst", "scf"],
            cwd=self.directory,
            check=True,
        )

    # =========================================================================
    # 4. SOC (スピン軌道相互作用) 計算の準備メソッド
    # =========================================================================

    def prepare_soc(self):
        """SOC計算用のサブディレクトリを用意し、必要なファイルを移動(move)する"""
        workdir = Path(self.directory)
        socdir = workdir / "SOC"
        socdir.mkdir(exist_ok=True)
        
        # バイナリと基本インプットを移動 (ver.0.0.3 で move に最適化)
        shutil.move(workdir / "pflapw", socdir / "pflapw")
        shutil.move(workdir / "lapwin", socdir / "lapwin")

        # FLcopy等によって生成された、末尾が 'scf' で終わる全中間ファイルを移動
        for f in workdir.iterdir():
            if f.is_file() and f.name.endswith("scf"):
                shutil.move(f, socdir / f.name)

        return socdir

    def prepare_soc_lapwin(self):
        """SOC計算用に lapwin ファイルの特定のパラメータ文字列を置換して書き換える"""
        lapwin = Path(self.directory) / "lapwin"
        text = lapwin.read_text()

        # P matrix 設定を F(False) から T(True) の詳細設定へ置換
        text = text.replace(
            "P matrix calculation:F, option:default",
            "P matrix calculation:T, option:set\n"
            "  valence-valence:T, if T set matrix type             !default:T\n"
            "    spin matrix:T                                     !default:F\n"
            "    orbital MT matrix:T                               !default:F\n"
            "    choose type:F, if T set species:                  !default:F\n"
            "  core-valence:F, if T set options"
        )

        # SOC (Second variational SOC) 設定を F から T へ置換
        text = text.replace(
            "Second variational SOC:F, option:default",
            "Second variational SOC:T, option:set\n"
            "  Spin rotation:W                               !Wigner/Euler default: W\n"
            "    theta(alph)   phi(beta)    (gamm)            !degree\n"
            "    0.0          0.0\n"
            "  Scaling of strength:F, if T factor for spec:  !default:F\n"
            "  Pre-factor:F"
        )

        # K点メッシュを高密度(50 50 50)へ置換（元設定の値を動的に指定して安全に置換）
        orig_kpts_str = f"  {self.kpts[0]}   {self.kpts[1]}   {self.kpts[2]}"
        text = text.replace(orig_kpts_str, "  50   50   50")

        # 初期スピン分極設定をオフにする
        text = text.replace(
            "Initial spin polarization:T",
            "Initial spin polarization:F"
        )

        lapwin.write_text(text)

    # =========================================================================
    # 5. Optics (光学特性) 計算用メソッド
    # =========================================================================

    def prepare_optics(self):
        """カレントディレクトリの 'opt' フォルダ内にある光学計算用ファイルを計算ディレクトリへコピーする"""
        root = Path.cwd()
        optdir = root / "opt"
        workdir = Path(self.directory)

        for f in optdir.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir / f.name)

    def run_xoptics(self):
        """光学特性計算プログラムである xoptics 実行ファイルを実行する"""
        subprocess.run(
            ["./xoptics"],
            cwd=self.directory,
            check=True,
        )
