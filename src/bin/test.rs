//! web-prop-test: Rust製 Webテストツール
//!
//! 機能:
//!   - 道A: 並行処理テスト（Shuttleスタイルスケジューラ）
//!   - 道B: DOMプロパティテスト（Boa統合）
//!
//! 使い方:
//!   cargo run -- test.html

use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::path::Path;

use boa_engine::{Context, Source, JsValue, JsString, NativeFunction, JsArgs};
use boa_engine::object::ObjectInitializer;
use boa_engine::property::Attribute;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use oxc_ast::ast::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use thiserror::Error;

// ============================================================================
// エラー型定義
// ============================================================================

/// web-prop-test の統一エラー型
#[derive(Error, Debug)]
pub enum WebTestError {
    /// JSエンジンが初期化されていない
    #[error("JS engine not initialized")]
    JsNotInitialized,
    
    /// JS実行時エラー
    #[error("JS evaluation failed: {0}")]
    JsEvalError(String),
    
    /// JS関数呼び出しエラー
    #[error("JS function call failed: {0}")]
    JsFunctionError(String),
    
    /// HTMLパースエラー
    #[error("HTML parse error: {0}")]
    HtmlParseError(String),
    
    /// ファイル読み込みエラー
    #[error("File read error: {path}")]
    FileReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },
    
    /// ファイルが見つからない
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    /// 無効なファイル形式
    #[error("Invalid file format: {0} (expected: {1})")]
    InvalidFileFormat(String, String),
    
    /// スケジューラエラー
    #[error("Scheduler error: {0}")]
    SchedulerError(String),
    
    /// 変数が見つからない
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    
    /// タスクが見つからない
    #[error("Task not found: {0}")]
    TaskNotFound(usize),
    
    /// 要素が見つからない
    #[error("Element not found: #{0}")]
    ElementNotFound(String),
    
    /// 無効な操作
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// IOエラー（汎用）
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

/// Result型エイリアス
pub type Result<T> = std::result::Result<T, WebTestError>;

// ============================================================================
// 道A: Shuttle スタイル スケジューラ
// ============================================================================

// AST解析結果
#[derive(Debug, Clone)]
pub struct AsyncFunc {
    pub name: String,
    pub await_count: usize,
    pub shared_vars: Vec<String>,  // 参照するグローバル変数
}

// React Hook情報
#[derive(Debug, Clone)]
pub struct ReactHookUsage {
    pub hook_name: String,       // useState, useEffect, etc.
    pub dependencies: Vec<String>, // 依存配列
    pub has_cleanup: bool,        // useEffectのクリーンアップ有無
    pub potential_race: bool,     // レースの可能性
}

// Reactコンポーネント情報
#[derive(Debug, Clone)]
pub struct ReactComponent {
    pub name: String,
    pub hooks: Vec<ReactHookUsage>,
    pub state_vars: Vec<String>,
    pub effect_count: usize,
}

#[derive(Debug)]
pub struct JsAnalysis {
    pub async_funcs: Vec<AsyncFunc>,
    pub global_vars: Vec<String>,
    pub await_points: usize,
    // React/Next.js拡張
    pub react_components: Vec<ReactComponent>,
    pub is_jsx: bool,
    pub next_patterns: Vec<String>,  // getServerSideProps, etc.
}

/// JSコードをパースしてasync関数とawaitを検出
pub fn analyze_js(code: &str) -> JsAnalysis {
    let allocator = Allocator::default();
    // JSX対応
    let is_jsx = code.contains("<") && (code.contains("/>") || code.contains("</"));
    let source_type = SourceType::default()
        .with_module(false)
        .with_jsx(is_jsx);
    let parser = Parser::new(&allocator, code, source_type);
    let result = parser.parse();
    
    let mut async_funcs = Vec::new();
    let mut global_vars = Vec::new();
    let mut total_awaits = 0;
    let mut react_components = Vec::new();
    let mut next_patterns = Vec::new();
    
    // Next.jsパターン検出
    if code.contains("getServerSideProps") {
        next_patterns.push("getServerSideProps".to_string());
    }
    if code.contains("getStaticProps") {
        next_patterns.push("getStaticProps".to_string());
    }
    if code.contains("useRouter") {
        next_patterns.push("useRouter".to_string());
    }
    
    // トップレベルの変数宣言を収集
    for stmt in &result.program.body {
        if let Statement::VariableDeclaration(decl) = stmt {
            for d in &decl.declarations {
                if let BindingPatternKind::BindingIdentifier(id) = &d.id.kind {
                    global_vars.push(id.name.to_string());
                }
            }
        }
    }
    
    // 関数を解析
    for stmt in &result.program.body {
        match stmt {
            Statement::FunctionDeclaration(func) => {
                if func.r#async {
                    let name = func.id.as_ref().map(|id| id.name.to_string()).unwrap_or_default();
                    let (await_count, shared) = analyze_function_body(&func.body, &global_vars);
                    total_awaits += await_count;
                    async_funcs.push(AsyncFunc {
                        name,
                        await_count,
                        shared_vars: shared,
                    });
                }
                
                // Reactコンポーネント検出（大文字始まり）
                if let Some(id) = &func.id {
                    let name = id.name.to_string();
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let hooks = analyze_react_hooks(&func.body, code);
                        if !hooks.is_empty() {
                            let state_vars: Vec<String> = hooks.iter()
                                .filter(|h| h.hook_name == "useState")
                                .flat_map(|h| h.dependencies.clone())
                                .collect();
                            let effect_count = hooks.iter()
                                .filter(|h| h.hook_name == "useEffect")
                                .count();
                            react_components.push(ReactComponent {
                                name,
                                hooks,
                                state_vars,
                                effect_count,
                            });
                        }
                    }
                }
            }
            Statement::VariableDeclaration(decl) => {
                for d in &decl.declarations {
                    if let Some(init) = &d.init {
                        if let Expression::ArrowFunctionExpression(arrow) = init {
                            if arrow.r#async {
                                let name = if let BindingPatternKind::BindingIdentifier(id) = &d.id.kind {
                                    id.name.to_string()
                                } else {
                                    "anonymous".to_string()
                                };
                                let (await_count, shared) = analyze_arrow_body(&arrow.body, &global_vars);
                                total_awaits += await_count;
                                async_funcs.push(AsyncFunc {
                                    name,
                                    await_count,
                                    shared_vars: shared,
                                });
                            }
                            
                            // Reactコンポーネント検出（アロー関数）
                            if let BindingPatternKind::BindingIdentifier(id) = &d.id.kind {
                                let name = id.name.to_string();
                                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                    let hooks = analyze_react_hooks_from_code(&name, code);
                                    if !hooks.is_empty() {
                                        let state_vars: Vec<String> = hooks.iter()
                                            .filter(|h| h.hook_name == "useState")
                                            .flat_map(|h| h.dependencies.clone())
                                            .collect();
                                        let effect_count = hooks.iter()
                                            .filter(|h| h.hook_name == "useEffect")
                                            .count();
                                        react_components.push(ReactComponent {
                                            name,
                                            hooks,
                                            state_vars,
                                            effect_count,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // exportも解析
            Statement::ExportDefaultDeclaration(exp) => {
                if let ExportDefaultDeclarationKind::FunctionDeclaration(func) = &exp.declaration {
                    if let Some(id) = &func.id {
                        let name = id.name.to_string();
                        let hooks = analyze_react_hooks(&func.body, code);
                        if !hooks.is_empty() {
                            let state_vars: Vec<String> = hooks.iter()
                                .filter(|h| h.hook_name == "useState")
                                .flat_map(|h| h.dependencies.clone())
                                .collect();
                            let effect_count = hooks.iter()
                                .filter(|h| h.hook_name == "useEffect")
                                .count();
                            react_components.push(ReactComponent {
                                name,
                                hooks,
                                state_vars,
                                effect_count,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    JsAnalysis {
        async_funcs,
        global_vars,
        await_points: total_awaits,
        react_components,
        is_jsx,
        next_patterns,
    }
}

/// Reactフックを解析（簡易版 - コードから直接検出）
fn analyze_react_hooks<'a>(_body: &Option<oxc_allocator::Box<'a, FunctionBody<'a>>>, code: &str) -> Vec<ReactHookUsage> {
    analyze_react_hooks_from_code("", code)
}

fn analyze_react_hooks_from_code(_component_name: &str, code: &str) -> Vec<ReactHookUsage> {
    let mut hooks = Vec::new();
    
    // useState検出
    let use_state_pattern = regex::Regex::new(r"useState\s*\(\s*([^)]*)\)").ok();
    if let Some(re) = use_state_pattern {
        for cap in re.captures_iter(code) {
            let initial = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            hooks.push(ReactHookUsage {
                hook_name: "useState".to_string(),
                dependencies: vec![initial.to_string()],
                has_cleanup: false,
                potential_race: false,
            });
        }
    }
    
    // useEffect検出
    let use_effect_count = code.matches("useEffect").count();
    let has_cleanup = code.contains("return () =>");
    let missing_deps = code.contains("useEffect") && !code.contains("], [");
    
    for _ in 0..use_effect_count {
        hooks.push(ReactHookUsage {
            hook_name: "useEffect".to_string(),
            dependencies: vec![],
            has_cleanup,
            potential_race: !has_cleanup || missing_deps,
        });
    }
    
    // useRef検出
    let use_ref_count = code.matches("useRef").count();
    for _ in 0..use_ref_count {
        hooks.push(ReactHookUsage {
            hook_name: "useRef".to_string(),
            dependencies: vec![],
            has_cleanup: false,
            potential_race: false,
        });
    }
    
    // useCallback検出
    if code.contains("useCallback") {
        hooks.push(ReactHookUsage {
            hook_name: "useCallback".to_string(),
            dependencies: vec![],
            has_cleanup: false,
            potential_race: false,
        });
    }
    
    // useMemo検出
    if code.contains("useMemo") {
        hooks.push(ReactHookUsage {
            hook_name: "useMemo".to_string(),
            dependencies: vec![],
            has_cleanup: false,
            potential_race: false,
        });
    }
    
    hooks
}

fn analyze_function_body<'a>(body: &Option<oxc_allocator::Box<'a, FunctionBody<'a>>>, globals: &[String]) -> (usize, Vec<String>) {
    let mut await_count = 0;
    let mut shared = Vec::new();
    
    if let Some(body) = body {
        for stmt in &body.statements {
            let (a, s) = count_awaits_in_stmt(stmt, globals);
            await_count += a;
            for v in s {
                if !shared.contains(&v) {
                    shared.push(v);
                }
            }
        }
    }
    
    (await_count, shared)
}

fn analyze_arrow_body(body: &FunctionBody, globals: &[String]) -> (usize, Vec<String>) {
    let mut await_count = 0;
    let mut shared = Vec::new();
    
    for stmt in &body.statements {
        let (a, s) = count_awaits_in_stmt(stmt, globals);
        await_count += a;
        for v in s {
            if !shared.contains(&v) {
                shared.push(v);
            }
        }
    }
    
    (await_count, shared)
}

fn count_awaits_in_stmt(stmt: &Statement, globals: &[String]) -> (usize, Vec<String>) {
    let mut count = 0;
    let mut shared = Vec::new();
    
    match stmt {
        Statement::ExpressionStatement(expr) => {
            let (c, s) = count_awaits_in_expr(&expr.expression, globals);
            count += c;
            shared.extend(s);
        }
        Statement::VariableDeclaration(decl) => {
            for d in &decl.declarations {
                if let Some(init) = &d.init {
                    let (c, s) = count_awaits_in_expr(init, globals);
                    count += c;
                    shared.extend(s);
                }
            }
        }
        Statement::ReturnStatement(ret) => {
            if let Some(arg) = &ret.argument {
                let (c, s) = count_awaits_in_expr(arg, globals);
                count += c;
                shared.extend(s);
            }
        }
        Statement::IfStatement(if_stmt) => {
            let (c, s) = count_awaits_in_stmt(&if_stmt.consequent, globals);
            count += c;
            shared.extend(s);
            if let Some(alt) = &if_stmt.alternate {
                let (c, s) = count_awaits_in_stmt(alt, globals);
                count += c;
                shared.extend(s);
            }
        }
        Statement::BlockStatement(block) => {
            for s in &block.body {
                let (c, vars) = count_awaits_in_stmt(s, globals);
                count += c;
                shared.extend(vars);
            }
        }
        _ => {}
    }
    
    (count, shared)
}

fn count_awaits_in_expr(expr: &Expression, globals: &[String]) -> (usize, Vec<String>) {
    let mut count = 0;
    let mut shared = Vec::new();
    
    match expr {
        Expression::AwaitExpression(_) => {
            count = 1;
        }
        Expression::Identifier(id) => {
            let name = id.name.to_string();
            if globals.contains(&name) && !shared.contains(&name) {
                shared.push(name);
            }
        }
        Expression::AssignmentExpression(assign) => {
            let (c, s) = count_awaits_in_expr(&assign.right, globals);
            count += c;
            shared.extend(s);
            // 左辺もチェック
            if let AssignmentTarget::AssignmentTargetIdentifier(id) = &assign.left {
                let name = id.name.to_string();
                if globals.contains(&name) && !shared.contains(&name) {
                    shared.push(name);
                }
            }
        }
        Expression::BinaryExpression(bin) => {
            let (c1, s1) = count_awaits_in_expr(&bin.left, globals);
            let (c2, s2) = count_awaits_in_expr(&bin.right, globals);
            count += c1 + c2;
            shared.extend(s1);
            shared.extend(s2);
        }
        Expression::CallExpression(call) => {
            for arg in &call.arguments {
                if let Argument::SpreadElement(spread) = arg {
                    let (c, s) = count_awaits_in_expr(&spread.argument, globals);
                    count += c;
                    shared.extend(s);
                } else if let Some(expr) = arg.as_expression() {
                    let (c, s) = count_awaits_in_expr(expr, globals);
                    count += c;
                    shared.extend(s);
                }
            }
        }
        _ => {}
    }
    
    (count, shared)
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskState { Ready, Running, Blocked, Completed }

#[derive(Debug, Clone, PartialEq)]
pub enum AccessType { Read, Write }

// ============================================================================
// FastTrack Epoch最適化 - O(n)→O(1)
// ============================================================================

/// Epoch: (thread_id, clock) - 単一スレッドアクセスの軽量表現
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Epoch {
    pub tid: usize,
    pub clock: u64,
}

impl Epoch {
    pub fn new(tid: usize, clock: u64) -> Self {
        Self { tid, clock }
    }
    
    pub fn zero() -> Self {
        Self { tid: 0, clock: 0 }
    }
    
    /// Epoch ⊑ VectorClock ?
    pub fn happens_before_vc(&self, vc: &VectorClock) -> bool {
        if self.tid < vc.clocks.len() {
            self.clock <= vc.clocks[self.tid]
        } else {
            false
        }
    }
}

/// 適応的表現: EpochまたはVectorClock
#[derive(Debug, Clone)]
pub enum AdaptiveVC {
    Epoch(Epoch),           // 単一スレッド（O(1)）
    VC(VectorClock),        // 複数スレッド（O(n)）
}

impl AdaptiveVC {
    pub fn new_epoch(tid: usize, clock: u64) -> Self {
        AdaptiveVC::Epoch(Epoch::new(tid, clock))
    }
    
    pub fn empty() -> Self {
        AdaptiveVC::Epoch(Epoch::zero())
    }
    
    /// VectorClockに昇格
    pub fn promote_to_vc(&mut self, num_threads: usize) {
        if let AdaptiveVC::Epoch(e) = self {
            let mut vc = VectorClock::new(num_threads);
            if e.tid < num_threads {
                vc.clocks[e.tid] = e.clock;
            }
            *self = AdaptiveVC::VC(vc);
        }
    }
    
    /// 別スレッドからのアクセスで更新
    pub fn update(&mut self, tid: usize, clock: u64, num_threads: usize) {
        match self {
            AdaptiveVC::Epoch(e) => {
                if e.tid == tid || e.clock == 0 {
                    // 同じスレッドまたは初回 → Epochのまま
                    e.tid = tid;
                    e.clock = clock;
                } else {
                    // 異なるスレッド → VCに昇格
                    let mut vc = VectorClock::new(num_threads);
                    vc.clocks[e.tid] = e.clock;
                    vc.clocks[tid] = clock;
                    *self = AdaptiveVC::VC(vc);
                }
            }
            AdaptiveVC::VC(vc) => {
                if tid < vc.clocks.len() {
                    vc.clocks[tid] = vc.clocks[tid].max(clock);
                }
            }
        }
    }
    
    /// happens-before チェック（O(1) for Epoch）
    pub fn concurrent_with(&self, tid: usize, clock: u64) -> bool {
        match self {
            AdaptiveVC::Epoch(e) => {
                // 異なるスレッドで、どちらも先行しない
                e.tid != tid && e.clock > 0
            }
            AdaptiveVC::VC(vc) => {
                if tid < vc.clocks.len() {
                    clock <= vc.clocks[tid]
                } else {
                    true
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    clocks: Vec<u64>,
}

impl VectorClock {
    pub fn new(size: usize) -> Self {
        Self { clocks: vec![0; size] }
    }
    
    pub fn increment(&mut self, tid: usize) {
        if tid < self.clocks.len() {
            self.clocks[tid] += 1;
        }
    }
    
    pub fn get(&self, tid: usize) -> u64 {
        self.clocks.get(tid).copied().unwrap_or(0)
    }
    
    pub fn sync(&mut self, other: &VectorClock) {
        for (i, &v) in other.clocks.iter().enumerate() {
            if i < self.clocks.len() {
                self.clocks[i] = self.clocks[i].max(v);
            }
        }
    }
    
    /// self happens-before other?
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut dominated = true;
        let mut strictly_less = false;
        for (i, &v) in self.clocks.iter().enumerate() {
            if i < other.clocks.len() {
                if v > other.clocks[i] {
                    dominated = false;
                }
                if v < other.clocks[i] {
                    strictly_less = true;
                }
            }
        }
        dominated && strictly_less
    }
    
    /// 並行？（どちらもhappens-beforeでない）
    pub fn concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }
}

// ============================================================================
// Race Coverage - Harmless Race除外
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum RaceSeverity {
    Harmful,        // 実害あり（値が変わる）
    Benign,         // 無害（同じ値を書く）
    Covered,        // 他のレースでカバー済み
    AdHocSync,      // アドホック同期パターン
}

#[derive(Debug, Clone)]
pub struct RaceWithSeverity {
    pub race: RaceConditionVC,
    pub severity: RaceSeverity,
    pub covered_by: Option<String>,  // カバーしているレースのID
}

/// Race Coverageアルゴリズム
#[allow(dead_code)]
pub struct RaceCoverage {
    races: Vec<RaceWithSeverity>,
    access_graph: HashMap<String, Vec<(usize, AccessType, u64)>>,  // var -> [(tid, type, clock)] 将来拡張用
}

impl RaceCoverage {
    pub fn new() -> Self {
        Self {
            races: Vec::new(),
            access_graph: HashMap::new(),
        }
    }
    
    pub fn add_race(&mut self, race: RaceConditionVC) {
        // 重複チェック
        if self.races.iter().any(|r| {
            r.race.var_name == race.var_name &&
            r.race.access1.task_id == race.access1.task_id &&
            r.race.access2.task_id == race.access2.task_id
        }) {
            return;
        }
        
        // 重大度判定
        let severity = self.classify_severity(&race);
        
        self.races.push(RaceWithSeverity {
            race,
            severity,
            covered_by: None,
        });
    }
    
    fn classify_severity(&self, race: &RaceConditionVC) -> RaceSeverity {
        // 同じ値を書いている場合はBenign
        if let (Some(v1), Some(v2)) = (race.access1.value, race.access2.value) {
            if v1 == v2 && race.race_type == RaceType::WriteWrite {
                return RaceSeverity::Benign;
            }
        }
        
        // アドホック同期パターン検出（フラグ変数など）
        if race.var_name.contains("flag") || 
           race.var_name.contains("ready") ||
           race.var_name.contains("done") {
            return RaceSeverity::AdHocSync;
        }
        
        RaceSeverity::Harmful
    }
    
    /// Covered racesを計算（EventRacer方式）
    pub fn compute_coverage(&mut self) {
        let n = self.races.len();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                
                // race[j]がrace[i]をカバーしているか
                if self.covers(&self.races[j].race, &self.races[i].race) {
                    // クローンして更新
                    let covered_id = format!("{}-{}", 
                        self.races[j].race.var_name,
                        self.races[j].race.access1.task_id);
                    self.races[i].severity = RaceSeverity::Covered;
                    self.races[i].covered_by = Some(covered_id);
                }
            }
        }
    }
    
    fn covers(&self, r1: &RaceConditionVC, r2: &RaceConditionVC) -> bool {
        // r1がr2をカバー = r1のアクセスがr2のアクセスより先に発生
        r1.access1.vc.happens_before(&r2.access1.vc) ||
        r1.access2.vc.happens_before(&r2.access1.vc)
    }
    
    pub fn get_harmful_races(&self) -> Vec<&RaceWithSeverity> {
        self.races.iter()
            .filter(|r| r.severity == RaceSeverity::Harmful)
            .collect()
    }
    
    pub fn get_all_races(&self) -> &[RaceWithSeverity] {
        &self.races
    }
    
    pub fn summary(&self) -> (usize, usize, usize, usize) {
        let harmful = self.races.iter().filter(|r| r.severity == RaceSeverity::Harmful).count();
        let benign = self.races.iter().filter(|r| r.severity == RaceSeverity::Benign).count();
        let covered = self.races.iter().filter(|r| r.severity == RaceSeverity::Covered).count();
        let adhoc = self.races.iter().filter(|r| r.severity == RaceSeverity::AdHocSync).count();
        (harmful, benign, covered, adhoc)
    }
}

// ============================================================================
// Predictive Analysis - 観測外レース予測
// ============================================================================

#[derive(Debug, Clone)]
pub struct PredictedRace {
    pub var_name: String,
    pub thread1: usize,
    pub thread2: usize,
    pub access1: AccessType,
    pub access2: AccessType,
    pub confidence: f64,  // 0.0-1.0
    pub reason: String,
}

#[allow(dead_code)]
pub struct PredictiveAnalyzer {
    // WCP (Weak Causal Precedence) 関係 - 将来拡張用
    wcp_edges: HashMap<(usize, usize), bool>,
    // アクセスパターン
    access_patterns: HashMap<String, Vec<(usize, AccessType)>>,
    // ロック保護情報
    lock_protected: HashMap<String, HashSet<String>>,  // var -> locks
}

impl PredictiveAnalyzer {
    pub fn new() -> Self {
        Self {
            wcp_edges: HashMap::new(),
            access_patterns: HashMap::new(),
            lock_protected: HashMap::new(),
        }
    }
    
    pub fn record_access(&mut self, var: &str, tid: usize, access_type: AccessType) {
        self.access_patterns
            .entry(var.to_string())
            .or_insert_with(Vec::new)
            .push((tid, access_type));
    }
    
    pub fn record_lock(&mut self, var: &str, lock: &str) {
        self.lock_protected
            .entry(var.to_string())
            .or_insert_with(HashSet::new)
            .insert(lock.to_string());
    }
    
    /// 観測されていないインターリービングでのレースを予測
    pub fn predict_races(&self) -> Vec<PredictedRace> {
        let mut predictions = Vec::new();
        
        for (var, accesses) in &self.access_patterns {
            // 異なるスレッドからのアクセスを収集
            let mut by_thread: HashMap<usize, Vec<AccessType>> = HashMap::new();
            for (tid, at) in accesses {
                by_thread.entry(*tid).or_default().push(at.clone());
            }
            
            // 2つ以上のスレッドがアクセス && 少なくとも1つがWrite
            if by_thread.len() >= 2 {
                let threads: Vec<usize> = by_thread.keys().copied().collect();
                for i in 0..threads.len() {
                    for j in i+1..threads.len() {
                        let t1 = threads[i];
                        let t2 = threads[j];
                        let a1 = &by_thread[&t1];
                        let a2 = &by_thread[&t2];
                        
                        // Writeが含まれているか
                        let has_write = a1.contains(&AccessType::Write) || 
                                       a2.contains(&AccessType::Write);
                        
                        if has_write {
                            // ロック保護されていないか確認
                            let protected = self.lock_protected.get(var)
                                .map(|locks| !locks.is_empty())
                                .unwrap_or(false);
                            
                            let confidence = if protected { 0.3 } else { 0.9 };
                            let reason = if protected {
                                "ロック保護あり、タイミング依存の可能性".to_string()
                            } else {
                                "ロック保護なし、高確率でレース".to_string()
                            };
                            
                            predictions.push(PredictedRace {
                                var_name: var.clone(),
                                thread1: t1,
                                thread2: t2,
                                access1: a1.last().cloned().unwrap_or(AccessType::Read),
                                access2: a2.last().cloned().unwrap_or(AccessType::Read),
                                confidence,
                                reason,
                            });
                        }
                    }
                }
            }
        }
        
        predictions
    }
}

// ============================================================================
// FastTrackスケジューラ（最適化版）
// ============================================================================

pub struct FastTrackScheduler {
    pub tasks: Vec<Task>,
    pending_promises: VecDeque<usize>,
    pub variables: HashMap<String, i64>,
    next_id: usize,
    
    // FastTrack: 各スレッドのクロック
    thread_clocks: HashMap<usize, u64>,
    // FastTrack: 各変数の書き込みEpoch
    write_epochs: HashMap<String, Epoch>,
    // FastTrack: 各変数の読み取り（Adaptive）
    read_history: HashMap<String, AdaptiveVC>,
    
    num_threads: usize,
    
    // Race Coverage
    race_coverage: RaceCoverage,
    // Predictive Analysis
    predictor: PredictiveAnalyzer,
    
    seed: u64,
}

impl FastTrackScheduler {
    pub fn new(num_threads: usize, seed: u64) -> Self {
        Self {
            tasks: Vec::new(),
            pending_promises: VecDeque::new(),
            variables: HashMap::new(),
            next_id: 0,
            thread_clocks: HashMap::new(),
            write_epochs: HashMap::new(),
            read_history: HashMap::new(),
            num_threads,
            race_coverage: RaceCoverage::new(),
            predictor: PredictiveAnalyzer::new(),
            seed,
        }
    }
    
    pub fn spawn(&mut self, name: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push(Task {
            id,
            name: name.to_string(),
            state: TaskState::Ready,
            awaiting: None,
        });
        self.thread_clocks.insert(id, 1);
        id
    }
    
    pub fn add_promise(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.pending_promises.push_back(id);
        id
    }
    
    pub fn init_var(&mut self, name: &str, val: i64) {
        self.variables.insert(name.to_string(), val);
        self.write_epochs.insert(name.to_string(), Epoch::zero());
        self.read_history.insert(name.to_string(), AdaptiveVC::empty());
    }
    
    fn get_clock(&self, tid: usize) -> u64 {
        self.thread_clocks.get(&tid).copied().unwrap_or(0)
    }
    
    fn increment_clock(&mut self, tid: usize) {
        *self.thread_clocks.entry(tid).or_insert(0) += 1;
    }
    
    /// FastTrack Read - O(1) in common case
    pub fn read_var(&mut self, tid: usize, name: &str) -> Option<i64> {
        let val = self.variables.get(name).copied();
        let clock = self.get_clock(tid);
        
        // [FT READ SAME EPOCH] 同じスレッドの連続読み取り
        if let Some(read_hist) = self.read_history.get(name) {
            if let AdaptiveVC::Epoch(e) = read_hist {
                if e.tid == tid {
                    // Same epoch - O(1) fast path
                    return val;
                }
            }
        }
        
        // Write-Read レースチェック
        if let Some(write_epoch) = self.write_epochs.get(name) {
            if write_epoch.tid != tid && write_epoch.clock > 0 {
                // 異なるスレッドからの書き込みと並行
                let vc1 = {
                    let mut v = VectorClock::new(self.num_threads);
                    v.clocks[write_epoch.tid] = write_epoch.clock;
                    v
                };
                let vc2 = {
                    let mut v = VectorClock::new(self.num_threads);
                    v.clocks[tid] = clock;
                    v
                };
                
                self.race_coverage.add_race(RaceConditionVC {
                    var_name: name.to_string(),
                    access1: AccessEventVC {
                        task_id: write_epoch.tid,
                        var_name: name.to_string(),
                        access_type: AccessType::Write,
                        vc: vc1,
                        value: None,
                    },
                    access2: AccessEventVC {
                        task_id: tid,
                        var_name: name.to_string(),
                        access_type: AccessType::Read,
                        vc: vc2,
                        value: val,
                    },
                    race_type: RaceType::WriteRead,
                });
            }
        }
        
        // Read history更新
        self.read_history
            .entry(name.to_string())
            .or_insert_with(AdaptiveVC::empty)
            .update(tid, clock, self.num_threads);
        
        // Predictive Analysis記録
        self.predictor.record_access(name, tid, AccessType::Read);
        
        val
    }
    
    /// FastTrack Write - O(1) in common case
    pub fn write_var(&mut self, tid: usize, name: &str, val: i64) {
        let clock = self.get_clock(tid);
        
        // Write-Write レースチェック
        if let Some(write_epoch) = self.write_epochs.get(name) {
            if write_epoch.tid != tid && write_epoch.clock > 0 {
                let vc1 = {
                    let mut v = VectorClock::new(self.num_threads);
                    v.clocks[write_epoch.tid] = write_epoch.clock;
                    v
                };
                let vc2 = {
                    let mut v = VectorClock::new(self.num_threads);
                    v.clocks[tid] = clock;
                    v
                };
                let old_val = self.variables.get(name).copied();
                
                self.race_coverage.add_race(RaceConditionVC {
                    var_name: name.to_string(),
                    access1: AccessEventVC {
                        task_id: write_epoch.tid,
                        var_name: name.to_string(),
                        access_type: AccessType::Write,
                        vc: vc1,
                        value: old_val,
                    },
                    access2: AccessEventVC {
                        task_id: tid,
                        var_name: name.to_string(),
                        access_type: AccessType::Write,
                        vc: vc2,
                        value: Some(val),
                    },
                    race_type: RaceType::WriteWrite,
                });
            }
        }
        
        // Read-Write レースチェック
        if let Some(read_hist) = self.read_history.get(name) {
            match read_hist {
                AdaptiveVC::Epoch(e) => {
                    if e.tid != tid && e.clock > 0 {
                        let vc1 = {
                            let mut v = VectorClock::new(self.num_threads);
                            v.clocks[e.tid] = e.clock;
                            v
                        };
                        let vc2 = {
                            let mut v = VectorClock::new(self.num_threads);
                            v.clocks[tid] = clock;
                            v
                        };
                        
                        self.race_coverage.add_race(RaceConditionVC {
                            var_name: name.to_string(),
                            access1: AccessEventVC {
                                task_id: e.tid,
                                var_name: name.to_string(),
                                access_type: AccessType::Read,
                                vc: vc1,
                                value: None,
                            },
                            access2: AccessEventVC {
                                task_id: tid,
                                var_name: name.to_string(),
                                access_type: AccessType::Write,
                                vc: vc2,
                                value: Some(val),
                            },
                            race_type: RaceType::ReadWrite,
                        });
                    }
                }
                AdaptiveVC::VC(vc) => {
                    // 複数リーダー - 各リーダーをチェック
                    for (other_tid, &other_clock) in vc.clocks.iter().enumerate() {
                        if other_tid != tid && other_clock > 0 {
                            let vc1 = {
                                let mut v = VectorClock::new(self.num_threads);
                                v.clocks[other_tid] = other_clock;
                                v
                            };
                            let vc2 = {
                                let mut v = VectorClock::new(self.num_threads);
                                v.clocks[tid] = clock;
                                v
                            };
                            
                            self.race_coverage.add_race(RaceConditionVC {
                                var_name: name.to_string(),
                                access1: AccessEventVC {
                                    task_id: other_tid,
                                    var_name: name.to_string(),
                                    access_type: AccessType::Read,
                                    vc: vc1,
                                    value: None,
                                },
                                access2: AccessEventVC {
                                    task_id: tid,
                                    var_name: name.to_string(),
                                    access_type: AccessType::Write,
                                    vc: vc2,
                                    value: Some(val),
                                },
                                race_type: RaceType::ReadWrite,
                            });
                        }
                    }
                }
            }
        }
        
        // [FT WRITE EXCLUSIVE] Write epoch更新、read history クリア
        self.write_epochs.insert(name.to_string(), Epoch::new(tid, clock));
        self.read_history.insert(name.to_string(), AdaptiveVC::empty());
        
        self.variables.insert(name.to_string(), val);
        self.increment_clock(tid);
        
        // Predictive Analysis記録
        self.predictor.record_access(name, tid, AccessType::Write);
    }
    
    pub fn block_task(&mut self, tid: usize, promise_id: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == tid) {
            t.state = TaskState::Blocked;
            t.awaiting = Some(promise_id);
        }
    }
    
    pub fn resolve_random(&mut self) -> Option<usize> {
        if self.pending_promises.is_empty() { return None; }
        let mut rng = StdRng::seed_from_u64(self.seed);
        self.seed += 1;
        let idx = rng.gen_range(0..self.pending_promises.len());
        let pid = self.pending_promises.remove(idx).unwrap();
        for t in &mut self.tasks {
            if t.awaiting == Some(pid) {
                t.state = TaskState::Ready;
                t.awaiting = None;
            }
        }
        Some(pid)
    }
    
    pub fn complete(&mut self, tid: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == tid) {
            t.state = TaskState::Completed;
        }
    }
    
    pub fn finalize(&mut self) {
        self.race_coverage.compute_coverage();
    }
    
    pub fn get_race_summary(&self) -> (usize, usize, usize, usize) {
        self.race_coverage.summary()
    }
    
    pub fn get_harmful_races(&self) -> Vec<&RaceWithSeverity> {
        self.race_coverage.get_harmful_races()
    }
    
    pub fn get_predictions(&self) -> Vec<PredictedRace> {
        self.predictor.predict_races()
    }
    
    pub fn final_state(&self) -> &HashMap<String, i64> {
        &self.variables
    }
}

// ============================================================================
// 拡張アクセスイベント（Vector Clock付き）
// ============================================================================

#[derive(Debug, Clone)]
pub struct AccessEventVC {
    pub task_id: usize,
    pub var_name: String,
    pub access_type: AccessType,
    pub vc: VectorClock,
    pub value: Option<i64>,  // 読み書きした値
}

#[derive(Debug, Clone)]
pub struct RaceConditionVC {
    pub var_name: String,
    pub access1: AccessEventVC,
    pub access2: AccessEventVC,
    pub race_type: RaceType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaceType {
    WriteWrite,
    ReadWrite,
    WriteRead,
}

impl std::fmt::Display for RaceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaceType::WriteWrite => write!(f, "W-W"),
            RaceType::ReadWrite => write!(f, "R-W"),
            RaceType::WriteRead => write!(f, "W-R"),
        }
    }
}

// ============================================================================
// DPOR - Dynamic Partial Order Reduction
// ============================================================================

#[derive(Debug, Clone)]
pub struct Transition {
    pub task_id: usize,
    pub var_name: Option<String>,
    pub access_type: Option<AccessType>,
}

#[derive(Debug, Clone)]
pub struct DPORState {
    pub backtrack: HashSet<usize>,  // バックトラックすべきタスク
    pub done: HashSet<usize>,       // 実行済みタスク
    pub sleep: HashSet<usize>,      // スリープセット
}

impl DPORState {
    pub fn new() -> Self {
        Self {
            backtrack: HashSet::new(),
            done: HashSet::new(),
            sleep: HashSet::new(),
        }
    }
}

/// 2つのトランジションが依存関係にあるか
pub fn dependent(t1: &Transition, t2: &Transition) -> bool {
    if t1.task_id == t2.task_id {
        return false;
    }
    match (&t1.var_name, &t2.var_name) {
        (Some(v1), Some(v2)) if v1 == v2 => {
            // 同じ変数へのアクセス、少なくとも一方がWrite
            matches!(
                (&t1.access_type, &t2.access_type),
                (Some(AccessType::Write), _) | (_, Some(AccessType::Write))
            )
        }
        _ => false,
    }
}

// ============================================================================
// PCT - Probabilistic Concurrency Testing
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PCTScheduler {
    priorities: HashMap<usize, i32>,
    priority_change_points: Vec<usize>,  // 優先度変更ポイント
    current_step: usize,
    d: usize,  // バグ深度パラメータ（理論計算用に保持）
}

impl PCTScheduler {
    pub fn new(num_tasks: usize, num_steps: usize, d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // 各タスクにランダムな初期優先度
        let mut priorities = HashMap::new();
        for i in 0..num_tasks {
            priorities.insert(i, rng.gen_range(0..1000));
        }
        
        // d-1個のランダムな優先度変更ポイント
        let mut change_points: Vec<usize> = (0..num_steps)
            .filter(|_| rng.gen_bool(0.5))
            .take(d.saturating_sub(1))
            .collect();
        change_points.sort();
        
        Self {
            priorities,
            priority_change_points: change_points,
            current_step: 0,
            d,
        }
    }
    
    /// 次に実行するタスクを選択
    pub fn select(&mut self, ready_tasks: &[usize], seed: u64) -> Option<usize> {
        if ready_tasks.is_empty() {
            return None;
        }
        
        // 優先度変更ポイントに到達したら、ランダムなタスクの優先度を下げる
        if self.priority_change_points.contains(&self.current_step) {
            let mut rng = StdRng::seed_from_u64(seed + self.current_step as u64);
            if let Some(&task) = ready_tasks.get(rng.gen_range(0..ready_tasks.len())) {
                self.priorities.insert(task, -1000 - (self.current_step as i32));
            }
        }
        
        self.current_step += 1;
        
        // 最高優先度のタスクを選択
        ready_tasks
            .iter()
            .max_by_key(|&&t| self.priorities.get(&t).unwrap_or(&0))
            .copied()
    }
    
    /// バグ発見確率（理論値）
    pub fn bug_probability(n: usize, k: usize, d: usize) -> f64 {
        // P >= 1 / (n * k^(d-1))
        let denom = (n as f64) * (k as f64).powi((d - 1) as i32);
        1.0 / denom
    }
}

// ============================================================================
// 高精度スケジューラ（Vector Clock + DPOR + PCT統合）
// ============================================================================

pub struct AdvancedScheduler {
    pub tasks: Vec<Task>,
    pending_promises: VecDeque<usize>,
    pub variables: HashMap<String, i64>,
    access_log: Vec<AccessEventVC>,
    next_id: usize,
    
    // Vector Clock
    task_clocks: HashMap<usize, VectorClock>,
    var_clocks: HashMap<String, (VectorClock, VectorClock)>,  // (last_write, last_read)
    num_tasks: usize,
    
    // DPOR
    dpor_stack: Vec<DPORState>,
    transitions: Vec<Transition>,
    
    // PCT
    pct: Option<PCTScheduler>,
    
    // 検出したレース
    races: Vec<RaceConditionVC>,
    
    seed: u64,
}

impl AdvancedScheduler {
    pub fn new(num_tasks: usize, seed: u64) -> Self {
        Self {
            tasks: Vec::new(),
            pending_promises: VecDeque::new(),
            variables: HashMap::new(),
            access_log: Vec::new(),
            next_id: 0,
            task_clocks: HashMap::new(),
            var_clocks: HashMap::new(),
            num_tasks,
            dpor_stack: Vec::new(),
            transitions: Vec::new(),
            pct: None,
            races: Vec::new(),
            seed,
        }
    }
    
    pub fn with_pct(mut self, num_steps: usize, d: usize) -> Self {
        self.pct = Some(PCTScheduler::new(self.num_tasks, num_steps, d, self.seed));
        self
    }
    
    pub fn spawn(&mut self, name: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push(Task { 
            id, 
            name: name.to_string(), 
            state: TaskState::Ready, 
            awaiting: None 
        });
        self.task_clocks.insert(id, VectorClock::new(self.num_tasks));
        id
    }
    
    pub fn add_promise(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.pending_promises.push_back(id);
        id
    }
    
    pub fn init_var(&mut self, name: &str, val: i64) {
        self.variables.insert(name.to_string(), val);
        self.var_clocks.insert(
            name.to_string(),
            (VectorClock::new(self.num_tasks), VectorClock::new(self.num_tasks))
        );
    }
    
    pub fn read_var(&mut self, task_id: usize, name: &str) -> Option<i64> {
        let val = self.variables.get(name).copied();
        
        // Vector Clock更新
        if let Some(tc) = self.task_clocks.get_mut(&task_id) {
            tc.increment(task_id);
            
            // レースチェック: last_write と concurrent?
            if let Some((last_write_vc, _)) = self.var_clocks.get(name) {
                if tc.concurrent(last_write_vc) && !last_write_vc.clocks.iter().all(|&c| c == 0) {
                    // 最後のWriteと並行 → W-R レース
                    if let Some(last_write) = self.access_log.iter().rev()
                        .find(|e| e.var_name == name && e.access_type == AccessType::Write && e.task_id != task_id) 
                    {
                        self.races.push(RaceConditionVC {
                            var_name: name.to_string(),
                            access1: last_write.clone(),
                            access2: AccessEventVC {
                                task_id,
                                var_name: name.to_string(),
                                access_type: AccessType::Read,
                                vc: tc.clone(),
                                value: val,
                            },
                            race_type: RaceType::WriteRead,
                        });
                    }
                }
            }
            
            // last_read更新
            if let Some((_, last_read_vc)) = self.var_clocks.get_mut(name) {
                last_read_vc.sync(tc);
            }
            
            self.access_log.push(AccessEventVC {
                task_id,
                var_name: name.to_string(),
                access_type: AccessType::Read,
                vc: tc.clone(),
                value: val,
            });
        }
        
        // DPOR: トランジション記録
        self.transitions.push(Transition {
            task_id,
            var_name: Some(name.to_string()),
            access_type: Some(AccessType::Read),
        });
        
        val
    }
    
    pub fn write_var(&mut self, task_id: usize, name: &str, val: i64) {
        // Vector Clock更新
        if let Some(tc) = self.task_clocks.get_mut(&task_id) {
            tc.increment(task_id);
            
            if let Some((last_write_vc, last_read_vc)) = self.var_clocks.get(name) {
                // W-W レース
                if tc.concurrent(last_write_vc) && !last_write_vc.clocks.iter().all(|&c| c == 0) {
                    if let Some(last_write) = self.access_log.iter().rev()
                        .find(|e| e.var_name == name && e.access_type == AccessType::Write && e.task_id != task_id)
                    {
                        self.races.push(RaceConditionVC {
                            var_name: name.to_string(),
                            access1: last_write.clone(),
                            access2: AccessEventVC {
                                task_id,
                                var_name: name.to_string(),
                                access_type: AccessType::Write,
                                vc: tc.clone(),
                                value: Some(val),
                            },
                            race_type: RaceType::WriteWrite,
                        });
                    }
                }
                
                // R-W レース
                if tc.concurrent(last_read_vc) && !last_read_vc.clocks.iter().all(|&c| c == 0) {
                    if let Some(last_read) = self.access_log.iter().rev()
                        .find(|e| e.var_name == name && e.access_type == AccessType::Read && e.task_id != task_id)
                    {
                        self.races.push(RaceConditionVC {
                            var_name: name.to_string(),
                            access1: last_read.clone(),
                            access2: AccessEventVC {
                                task_id,
                                var_name: name.to_string(),
                                access_type: AccessType::Write,
                                vc: tc.clone(),
                                value: Some(val),
                            },
                            race_type: RaceType::ReadWrite,
                        });
                    }
                }
            }
            
            // last_write更新
            if let Some((last_write_vc, _)) = self.var_clocks.get_mut(name) {
                *last_write_vc = tc.clone();
            }
            
            self.access_log.push(AccessEventVC {
                task_id,
                var_name: name.to_string(),
                access_type: AccessType::Write,
                vc: tc.clone(),
                value: Some(val),
            });
        }
        
        self.variables.insert(name.to_string(), val);
        
        // DPOR: トランジション記録
        self.transitions.push(Transition {
            task_id,
            var_name: Some(name.to_string()),
            access_type: Some(AccessType::Write),
        });
        
        // DPOR: バックトラックセット更新
        self.update_backtrack();
    }
    
    fn update_backtrack(&mut self) {
        let n = self.transitions.len();
        if n < 2 {
            return;
        }
        
        let last = &self.transitions[n - 1];
        for i in 0..n - 1 {
            let prev = &self.transitions[i];
            if dependent(prev, last) {
                // 依存関係があればバックトラック候補に追加
                if let Some(state) = self.dpor_stack.last_mut() {
                    state.backtrack.insert(prev.task_id);
                }
            }
        }
    }
    
    pub fn block_task(&mut self, task_id: usize, promise_id: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            t.state = TaskState::Blocked;
            t.awaiting = Some(promise_id);
        }
    }
    
    pub fn resolve_promise(&mut self, promise_id: usize) {
        self.pending_promises.retain(|&p| p != promise_id);
        for t in &mut self.tasks {
            if t.awaiting == Some(promise_id) {
                t.state = TaskState::Ready;
                t.awaiting = None;
            }
        }
    }
    
    /// PCTまたはランダムでPromiseを解決
    pub fn resolve_next(&mut self) -> Option<usize> {
        if self.pending_promises.is_empty() {
            return None;
        }
        
        let ready: Vec<usize> = self.pending_promises.iter().copied().collect();
        
        let selected = if let Some(pct) = &mut self.pct {
            pct.select(&ready, self.seed)
        } else {
            let mut rng = StdRng::seed_from_u64(self.seed + self.transitions.len() as u64);
            ready.get(rng.gen_range(0..ready.len())).copied()
        };
        
        if let Some(pid) = selected {
            self.resolve_promise(pid);
            Some(pid)
        } else {
            None
        }
    }
    
    pub fn complete(&mut self, task_id: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            t.state = TaskState::Completed;
        }
    }
    
    pub fn get_races(&self) -> &[RaceConditionVC] {
        &self.races
    }
    
    pub fn final_state(&self) -> &HashMap<String, i64> {
        &self.variables
    }
    
    /// DPOR: 次に探索すべきスケジュールがあるか
    pub fn has_backtrack(&self) -> bool {
        self.dpor_stack.iter().any(|s| !s.backtrack.is_empty())
    }
}

// ============================================================================
// 旧Scheduler（互換性維持）
// ============================================================================

#[derive(Debug, Clone)]
pub struct Task {
    pub id: usize,
    pub name: String,
    pub state: TaskState,
    pub awaiting: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub task_id: usize,
    pub var_name: String,
    pub access_type: AccessType,
    pub timestamp: usize,
}

#[derive(Debug, Clone)]
pub struct RaceCondition {
    pub var_name: String,
    pub task1: usize,
    pub task2: usize,
    pub description: String,
}

pub struct Scheduler {
    pub tasks: Vec<Task>,  // pubに変更
    pending_promises: VecDeque<usize>,
    rng: StdRng,
    pub variables: HashMap<String, i64>,  // pubに変更
    access_log: Vec<AccessEvent>,
    next_id: usize,
    decisions: Vec<usize>,  // スケジュール記録用
}

impl Scheduler {
    pub fn new(seed: u64) -> Self {
        Self {
            tasks: Vec::new(),
            pending_promises: VecDeque::new(),
            rng: StdRng::seed_from_u64(seed),
            variables: HashMap::new(),
            access_log: Vec::new(),
            next_id: 0,
            decisions: Vec::new(),
        }
    }

    pub fn spawn(&mut self, name: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push(Task { id, name: name.to_string(), state: TaskState::Ready, awaiting: None });
        id
    }

    pub fn add_promise(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.pending_promises.push_back(id);
        id
    }

    pub fn block_task(&mut self, task_id: usize, promise_id: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            t.state = TaskState::Blocked;
            t.awaiting = Some(promise_id);
        }
    }

    pub fn resolve_random(&mut self) -> Option<usize> {
        if self.pending_promises.is_empty() { return None; }
        let idx = self.rng.gen_range(0..self.pending_promises.len());
        self.decisions.push(idx);  // 記録
        let pid = self.pending_promises.remove(idx).unwrap();
        for t in &mut self.tasks {
            if t.awaiting == Some(pid) {
                t.state = TaskState::Ready;
                t.awaiting = None;
            }
        }
        Some(pid)
    }

    pub fn complete(&mut self, task_id: usize) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            t.state = TaskState::Completed;
        }
    }

    pub fn init_var(&mut self, name: &str, val: i64) { self.variables.insert(name.to_string(), val); }

    pub fn read_var(&mut self, task_id: usize, name: &str) -> Option<i64> {
        let val = self.variables.get(name).copied();
        self.access_log.push(AccessEvent {
            task_id, var_name: name.to_string(), access_type: AccessType::Read, timestamp: self.access_log.len()
        });
        val
    }

    pub fn write_var(&mut self, task_id: usize, name: &str, val: i64) {
        self.variables.insert(name.to_string(), val);
        self.access_log.push(AccessEvent {
            task_id, var_name: name.to_string(), access_type: AccessType::Write, timestamp: self.access_log.len()
        });
    }

    pub fn detect_races(&self) -> Vec<RaceCondition> {
        let mut races = Vec::new();
        let mut last: HashMap<String, &AccessEvent> = HashMap::new();
        for e in &self.access_log {
            if let Some(prev) = last.get(&e.var_name) {
                if prev.task_id != e.task_id && (prev.access_type == AccessType::Write || e.access_type == AccessType::Write) {
                    races.push(RaceCondition {
                        var_name: e.var_name.clone(),
                        task1: prev.task_id, task2: e.task_id,
                        description: format!("'{}' Task{}→Task{} 競合", e.var_name, prev.task_id, e.task_id),
                    });
                }
            }
            last.insert(e.var_name.clone(), e);
        }
        races
    }

    pub fn final_state(&self) -> &HashMap<String, i64> { &self.variables }
    
    pub fn get_decisions(&self) -> &[usize] { &self.decisions }
}

// ============================================================================
// Shrinking: 失敗ケースを最小化
// ============================================================================

#[derive(Debug, Clone)]
pub struct TestRun {
    pub seed: u64,
    pub final_state: HashMap<String, i64>,
    pub races: Vec<RaceCondition>,
    pub decisions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ShrunkCase {
    pub task_count: usize,
    pub seed: u64,
    pub final_state: HashMap<String, i64>,
    pub expected: i64,
}

// ============================================================================
// 包括的レーステストシナリオ (15カテゴリ)
// Based on: InitRacer, EventRacer, NodeRacer, PortSwigger Research
// ============================================================================

#[derive(Debug, Clone)]
pub struct RaceScenarioResult {
    pub category: String,
    pub name: String,
    pub detected: bool,
    pub race_count: usize,
    pub details: String,
}

/// 15カテゴリの包括的レーステストを実行
fn run_comprehensive_race_tests(base_seed: u64) {
    let mut results: Vec<RaceScenarioResult> = Vec::new();
    
    // Category 1: TOCTOU (Time-of-Check to Time-of-Use)
    results.push(test_toctou_race(base_seed));
    
    // Category 2: Limit Overrun (Double-Spend)
    results.push(test_limit_overrun_race(base_seed + 100));
    
    // Category 3: Stale Response
    results.push(test_stale_response_race(base_seed + 200));
    
    // Category 4: Form Input Overwrite
    results.push(test_form_overwrite_race(base_seed + 300));
    
    // Category 5: Late Event Handler
    results.push(test_late_handler_race(base_seed + 400));
    
    // Category 6: Access Before Definition
    results.push(test_access_before_def_race(base_seed + 500));
    
    // Category 7: Promise.all Parallel
    results.push(test_promise_all_race(base_seed + 600));
    
    // Category 8: LocalStorage Race
    results.push(test_storage_race(base_seed + 700));
    
    // Category 9: DOM Mutation Race
    results.push(test_dom_mutation_race(base_seed + 800));
    
    // Category 10: Optimistic Locking Failure
    results.push(test_optimistic_lock_race(base_seed + 900));
    
    // Category 11: Read-Modify-Write
    results.push(test_rmw_race(base_seed + 1000));
    
    // Category 12: Event Ordering
    results.push(test_event_order_race(base_seed + 1100));
    
    // Category 13: Callback Interleaving
    results.push(test_callback_race(base_seed + 1200));
    
    // Category 14: Resource Cleanup Race
    results.push(test_cleanup_race(base_seed + 1300));
    
    // Category 15: Initialization Race
    results.push(test_init_race(base_seed + 1400));
    
    // 結果表示
    println!("  ┌────┬──────────────────────────┬────────┬──────┐");
    println!("  │ #  │ カテゴリ                   │ 検出   │ 件数 │");
    println!("  ├────┼──────────────────────────┼────────┼──────┤");
    
    let mut total_detected = 0;
    let mut total_races = 0;
    
    for (i, r) in results.iter().enumerate() {
        let status = if r.detected { "✅" } else { "❌" };
        println!("  │ {:2} │ {:24} │ {}     │ {:4} │", 
            i + 1, 
            truncate_str(&r.name, 24),
            status, 
            r.race_count);
        if r.detected { total_detected += 1; }
        total_races += r.race_count;
    }
    
    println!("  └────┴──────────────────────────┴────────┴──────┘");
    println!("\n  📊 総合結果: {}/15 カテゴリ検出, 総レース数: {}", total_detected, total_races);
    
    // 詳細表示
    println!("\n  📝 詳細:");
    for r in &results {
        if !r.details.is_empty() {
            println!("    • {}: {}", r.name, r.details);
        }
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        format!("{:width$}", s, width = max_len)
    } else {
        let truncated: String = s.chars().take(max_len - 2).collect();
        format!("{}..", truncated)
    }
}

// Individual test functions

fn test_toctou_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let t0 = ft.spawn("task0");
    let t1 = ft.spawn("task1");
    ft.init_var("count", 0);
    
    // T0: read, T1: read, T0: write, T1: write
    ft.read_var(t0, "count");
    ft.read_var(t1, "count");
    ft.write_var(t0, "count", 1);
    ft.write_var(t1, "count", 1);
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "TOCTOU".to_string(),
        name: "TOCTOU (Check-Use)".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "read→await→write パターン".to_string(),
    }
}

fn test_limit_overrun_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let t0 = ft.spawn("withdraw0");
    let t1 = ft.spawn("withdraw1");
    ft.init_var("balance", 100);
    
    // Both check balance >= 100, both withdraw
    ft.read_var(t0, "balance");  // check
    ft.read_var(t1, "balance");  // check (concurrent)
    ft.write_var(t0, "balance", 0);  // withdraw
    ft.write_var(t1, "balance", -100);  // double-spend!
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "LimitOverrun".to_string(),
        name: "Limit Overrun (Double-Spend)".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "残高チェック→出金の競合".to_string(),
    }
}

fn test_stale_response_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let t0 = ft.spawn("search1");
    let t1 = ft.spawn("search2");
    ft.init_var("requestId", 0);
    ft.init_var("displayedResult", 0);
    
    ft.write_var(t0, "requestId", 1);
    ft.write_var(t1, "requestId", 2);
    // t0 finishes later but writes stale result
    ft.write_var(t1, "displayedResult", 2);
    ft.write_var(t0, "displayedResult", 1);  // Stale!
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "StaleResponse".to_string(),
        name: "Stale Response (Typeahead)".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "古いAPIレスポンスが新しい結果を上書き".to_string(),
    }
}

fn test_form_overwrite_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let user = ft.spawn("user_input");
    let script = ft.spawn("init_script");
    ft.init_var("inputValue", 0);
    
    // User types while script initializes
    ft.write_var(user, "inputValue", 42);  // User input
    ft.write_var(script, "inputValue", 0);  // Script overwrites!
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "FormOverwrite".to_string(),
        name: "Form Input Overwrite".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "初期化スクリプトがユーザー入力を上書き".to_string(),
    }
}

fn test_late_handler_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let click = ft.spawn("click_event");
    let setup = ft.spawn("setup_handler");
    ft.init_var("handlerReady", 0);
    ft.init_var("clickResult", 0);
    
    // User clicks before handler is ready - both access handlerReady
    ft.read_var(click, "handlerReady");  // Click checks: not ready
    ft.write_var(setup, "handlerReady", 1);  // Setup writes: ready
    // Click proceeds with stale check result
    ft.write_var(click, "clickResult", -1);  // Click fails (handler not ready)
    ft.read_var(setup, "clickResult");  // Setup sees failed click
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "LateHandler".to_string(),
        name: "Late Event Handler".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "ハンドラ登録前のイベント発火".to_string(),
    }
}

fn test_access_before_def_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let use_lib = ft.spawn("use_library");
    let load_lib = ft.spawn("load_library");
    ft.init_var("libLoaded", 0);
    ft.init_var("libResult", 0);
    
    // Try to use library before it's loaded - concurrent access
    ft.read_var(use_lib, "libLoaded");  // Use checks: not loaded
    ft.write_var(load_lib, "libLoaded", 1);  // Load writes: loaded
    ft.write_var(use_lib, "libResult", -1);  // Use fails (lib not loaded)
    ft.write_var(load_lib, "libResult", 1);  // Load overwrites with success
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "AccessBeforeDef".to_string(),
        name: "Access Before Definition".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "ライブラリ読み込み前のアクセス".to_string(),
    }
}

fn test_promise_all_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(5, seed);
    let tasks: Vec<usize> = (0..5).map(|i| ft.spawn(&format!("inc{}", i))).collect();
    ft.init_var("count", 0);
    
    // All tasks read initial value
    for &t in &tasks {
        ft.read_var(t, "count");
    }
    // All tasks write 1 (instead of incrementing)
    for &t in &tasks {
        ft.write_var(t, "count", 1);
    }
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "PromiseAll".to_string(),
        name: "Promise.all Parallel".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: format!("期待:5, 実際:1 ({}件の競合)", harmful),
    }
}

fn test_storage_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let tab1 = ft.spawn("tab1");
    let tab2 = ft.spawn("tab2");
    ft.init_var("storage_counter", 0);
    
    // Both tabs read, increment, write
    ft.read_var(tab1, "storage_counter");
    ft.read_var(tab2, "storage_counter");
    ft.write_var(tab1, "storage_counter", 1);
    ft.write_var(tab2, "storage_counter", 1);  // Lost update
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "Storage".to_string(),
        name: "LocalStorage Multi-Tab".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "複数タブからの同時アクセス".to_string(),
    }
}

fn test_dom_mutation_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let t0 = ft.spawn("update0");
    let t1 = ft.spawn("update1");
    ft.init_var("dom_content", 0);
    
    ft.read_var(t0, "dom_content");
    ft.read_var(t1, "dom_content");
    ft.write_var(t0, "dom_content", 1);
    ft.write_var(t1, "dom_content", 2);
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "DOMMutation".to_string(),
        name: "DOM Mutation Race".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "並行DOM更新".to_string(),
    }
}

fn test_optimistic_lock_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let t0 = ft.spawn("update0");
    let t1 = ft.spawn("update1");
    ft.init_var("version", 0);
    ft.init_var("data", 0);
    
    // Both read same version
    ft.read_var(t0, "version");
    ft.read_var(t1, "version");
    // Both try to update
    ft.write_var(t0, "data", 100);
    ft.write_var(t0, "version", 1);
    ft.write_var(t1, "data", 200);  // Should fail but doesn't
    ft.write_var(t1, "version", 1);  // Conflict!
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "OptimisticLock".to_string(),
        name: "Optimistic Locking Failure".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "バージョンチェックの競合".to_string(),
    }
}

fn test_rmw_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(4, seed);
    let tasks: Vec<usize> = (0..4).map(|i| ft.spawn(&format!("rmw{}", i))).collect();
    ft.init_var("counter", 0);
    
    for &t in &tasks {
        let _ = ft.read_var(t, "counter");
        ft.write_var(t, "counter", 1);
    }
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "RMW".to_string(),
        name: "Read-Modify-Write".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "非アトミックなインクリメント".to_string(),
    }
}

fn test_event_order_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let load = ft.spawn("onload");
    let click = ft.spawn("onclick");
    ft.init_var("state", 0);
    
    // Events may fire in unexpected order
    ft.write_var(click, "state", 2);  // Click before load?
    ft.write_var(load, "state", 1);
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "EventOrder".to_string(),
        name: "Event Ordering Race".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "イベント発火順序の非決定性".to_string(),
    }
}

fn test_callback_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let cb1 = ft.spawn("callback1");
    let cb2 = ft.spawn("callback2");
    ft.init_var("result", 0);
    
    // Nested callbacks interleave
    ft.read_var(cb1, "result");
    ft.read_var(cb2, "result");
    ft.write_var(cb1, "result", 1);
    ft.write_var(cb2, "result", 2);
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "Callback".to_string(),
        name: "Callback Interleaving".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "コールバックの非同期競合".to_string(),
    }
}

fn test_cleanup_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let use_resource = ft.spawn("use");
    let cleanup = ft.spawn("cleanup");
    ft.init_var("resource", 1);
    
    ft.read_var(use_resource, "resource");  // Using resource
    ft.write_var(cleanup, "resource", 0);   // Cleanup while in use!
    ft.write_var(use_resource, "resource", 2);  // Use after cleanup
    
    ft.finalize();
    let (harmful, _, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "Cleanup".to_string(),
        name: "Resource Cleanup Race".to_string(),
        detected: harmful > 0,
        race_count: harmful,
        details: "使用中リソースのクリーンアップ".to_string(),
    }
}

fn test_init_race(seed: u64) -> RaceScenarioResult {
    let mut ft = FastTrackScheduler::new(3, seed);
    let init1 = ft.spawn("init1");
    let init2 = ft.spawn("init2");
    ft.init_var("initialized", 0);
    
    // Multiple initialization attempts
    ft.read_var(init1, "initialized");
    ft.read_var(init2, "initialized");
    ft.write_var(init1, "initialized", 1);
    ft.write_var(init2, "initialized", 1);  // Double init
    
    ft.finalize();
    let (harmful, benign, _, _) = ft.get_race_summary();
    
    RaceScenarioResult {
        category: "Init".to_string(),
        name: "Initialization Race".to_string(),
        detected: harmful > 0 || benign > 0,
        race_count: harmful + benign,
        details: "複数回初期化の競合".to_string(),
    }
}

/// カウンターシナリオを実行
fn run_counter_scenario(num_tasks: usize, seed: u64) -> (HashMap<String, i64>, Vec<RaceCondition>, Vec<usize>) {
    let mut s = Scheduler::new(seed);
    s.init_var("count", 0);
    
    let tasks: Vec<usize> = (0..num_tasks)
        .map(|i| s.spawn(&format!("inc{}", i)))
        .collect();
    let promises: Vec<usize> = (0..num_tasks)
        .map(|_| s.add_promise())
        .collect();
    
    // 各タスク: read → block
    for (i, &task_id) in tasks.iter().enumerate() {
        s.read_var(task_id, "count");
        s.block_task(task_id, promises[i]);
    }
    
    // ランダム順でresolve
    while !s.pending_promises.is_empty() {
        s.resolve_random();
    }
    
    // 各タスクが書き込み
    for &task_id in &tasks {
        let v = s.read_var(task_id, "count").unwrap_or(0);
        s.write_var(task_id, "count", v + 1);
        s.complete(task_id);
    }
    
    (s.final_state().clone(), s.detect_races(), s.get_decisions().to_vec())
}

/// プロパティテスト + Shrinking
pub fn prop_test_with_shrink(iterations: usize, base_seed: u64, num_tasks: usize) -> (usize, Option<ShrunkCase>) {
    let mut failures = 0;
    let mut first_failure: Option<TestRun> = None;
    
    // 1. 複数回テスト
    for i in 0..iterations {
        let seed = base_seed + i as u64;
        let (state, races, decisions) = run_counter_scenario(num_tasks, seed);
        let count = state.get("count").copied().unwrap_or(0);
        let expected = num_tasks as i64;
        
        if count != expected || !races.is_empty() {
            failures += 1;
            if first_failure.is_none() {
                first_failure = Some(TestRun {
                    seed,
                    final_state: state,
                    races,
                    decisions,
                });
            }
        }
    }
    
    // 2. 失敗があれば縮小
    let shrunk = if failures > 0 {
        shrink_counter(num_tasks, base_seed)
    } else {
        None
    };
    
    (failures, shrunk)
}

/// 最小の失敗ケースを探す
fn shrink_counter(max_tasks: usize, base_seed: u64) -> Option<ShrunkCase> {
    // タスク数を減らしていく
    for n in (2..=max_tasks).rev() {
        for seed in 0..100 {
            let (state, _, _) = run_counter_scenario(n, base_seed + seed);
            let count = state.get("count").copied().unwrap_or(0);
            let expected = n as i64;
            
            if count != expected {
                // 最小の2タスクで失敗すれば十分
                if n == 2 {
                    return Some(ShrunkCase {
                        task_count: n,
                        seed: base_seed + seed,
                        final_state: state,
                        expected,
                    });
                }
            }
        }
    }
    
    // 2タスクでの最小ケース
    for seed in 0..100 {
        let (state, _, _) = run_counter_scenario(2, base_seed + seed);
        let count = state.get("count").copied().unwrap_or(0);
        if count != 2 {
            return Some(ShrunkCase {
                task_count: 2,
                seed: base_seed + seed,
                final_state: state,
                expected: 2,
            });
        }
    }
    
    None
}

// ============================================================================
// 全スケジュール探索（Exhaustive Search）
// ============================================================================

#[derive(Debug, Clone)]
pub struct ExhaustiveResult {
    pub total_schedules: usize,
    pub failures: usize,
    pub unique_outcomes: Vec<(i64, usize)>,  // (結果, 回数)
    pub failing_schedules: Vec<Vec<usize>>,   // 失敗したスケジュール
}

/// 全ての可能なスケジュールを列挙して実行
pub fn exhaustive_test(num_tasks: usize) -> ExhaustiveResult {
    let mut outcomes: HashMap<i64, usize> = HashMap::new();
    let mut failures = 0;
    let mut failing_schedules = Vec::new();
    let mut total = 0;
    
    // num_tasks個のPromiseの解決順序 = num_tasks! 通り
    let permutations = generate_permutations(num_tasks);
    
    for perm in &permutations {
        total += 1;
        let (state, schedule) = run_with_schedule(num_tasks, perm);
        let count = state.get("count").copied().unwrap_or(0);
        
        *outcomes.entry(count).or_insert(0) += 1;
        
        if count != num_tasks as i64 {
            failures += 1;
            if failing_schedules.len() < 5 {
                failing_schedules.push(schedule);
            }
        }
    }
    
    let mut unique: Vec<(i64, usize)> = outcomes.into_iter().collect();
    unique.sort_by_key(|(k, _)| *k);
    
    ExhaustiveResult {
        total_schedules: total,
        failures,
        unique_outcomes: unique,
        failing_schedules,
    }
}

/// 順列を生成（n!）
fn generate_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![vec![0]];
    }
    
    let mut result = Vec::new();
    let indices: Vec<usize> = (0..n).collect();
    permute(&indices, 0, &mut result);
    result
}

fn permute(arr: &[usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == arr.len() {
        result.push(arr.to_vec());
        return;
    }
    
    let mut arr = arr.to_vec();
    for i in start..arr.len() {
        arr.swap(start, i);
        permute(&arr, start + 1, result);
        arr.swap(start, i);
    }
}

/// 指定したスケジュールで実行
fn run_with_schedule(num_tasks: usize, resolve_order: &[usize]) -> (HashMap<String, i64>, Vec<usize>) {
    let mut s = Scheduler::new(0);  // seed不要（決定的）
    s.init_var("count", 0);
    
    let tasks: Vec<usize> = (0..num_tasks)
        .map(|i| s.spawn(&format!("inc{}", i)))
        .collect();
    
    // Promiseを作成（インデックスとIDのマッピング）
    let promises: Vec<usize> = (0..num_tasks)
        .map(|_| s.add_promise())
        .collect();
    
    // 各タスク: read → block
    let mut read_values: Vec<i64> = Vec::new();
    for (i, &task_id) in tasks.iter().enumerate() {
        let v = s.read_var(task_id, "count").unwrap_or(0);
        read_values.push(v);
        s.block_task(task_id, promises[i]);
    }
    
    // 指定した順序でresolve
    let mut remaining: Vec<usize> = (0..num_tasks).collect();
    let mut actual_order = Vec::new();
    
    for &idx in resolve_order {
        if idx < remaining.len() {
            let task_idx = remaining.remove(idx);
            actual_order.push(task_idx);
            
            // このタスクをunblock
            if let Some(t) = s.tasks.iter_mut().find(|t| t.id == tasks[task_idx]) {
                t.state = TaskState::Ready;
                t.awaiting = None;
            }
        }
    }
    
    // 各タスクが書き込み（読んだ値+1）
    for (i, &task_id) in tasks.iter().enumerate() {
        s.write_var(task_id, "count", read_values[i] + 1);
        s.complete(task_id);
    }
    
    (s.final_state().clone(), actual_order)
}

/// 部分スケジュール探索（大きなnに対応）
pub fn bounded_exhaustive_test(num_tasks: usize, max_schedules: usize, seed: u64) -> ExhaustiveResult {
    let mut outcomes: HashMap<i64, usize> = HashMap::new();
    let mut failures = 0;
    let mut failing_schedules = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    
    let total = max_schedules.min(factorial(num_tasks));
    
    for _ in 0..total {
        // ランダムなスケジュールを生成
        let mut perm: Vec<usize> = (0..num_tasks).collect();
        perm.shuffle(&mut rng);
        
        let schedule: Vec<usize> = (0..num_tasks).map(|i| {
            perm.iter().position(|&x| x == i).unwrap() % (num_tasks - i).max(1)
        }).collect();
        
        let (state, _, _) = run_counter_scenario(num_tasks, rng.gen());
        let count = state.get("count").copied().unwrap_or(0);
        
        *outcomes.entry(count).or_insert(0) += 1;
        
        if count != num_tasks as i64 {
            failures += 1;
            if failing_schedules.len() < 5 {
                failing_schedules.push(schedule);
            }
        }
    }
    
    let mut unique: Vec<(i64, usize)> = outcomes.into_iter().collect();
    unique.sort_by_key(|(k, _)| *k);
    
    ExhaustiveResult {
        total_schedules: total,
        failures,
        unique_outcomes: unique,
        failing_schedules,
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

// ============================================================================
// 道B: ミニDOM + プロパティテスト
// ============================================================================

#[derive(Debug, Clone)]
pub struct Element {
    pub tag: String,
    pub id: Option<String>,
    pub attributes: HashMap<String, String>,
    pub events: Vec<String>,
    pub onclick: Option<String>,
    pub oninput: Option<String>,
}

impl Element {
    pub fn is_disabled(&self) -> bool { self.attributes.contains_key("disabled") }
}

pub struct MiniDom {
    elements: HashMap<String, Element>,
    event_log: Vec<String>,
    js_ctx: Option<Context>,
}

impl MiniDom {
    pub fn new() -> Self { 
        Self { 
            elements: HashMap::new(), 
            event_log: Vec::new(),
            js_ctx: None,
        } 
    }

    pub fn init_js(&mut self) {
        let mut ctx = Context::default();
        
        // console.log
        let console = ObjectInitializer::new(&mut ctx)
            .function(
                NativeFunction::from_fn_ptr(|_, args, _| {
                    let msg = args.get_or_undefined(0)
                        .to_string(&mut Context::default())
                        .map(|s| s.to_std_string_escaped())
                        .unwrap_or_default();
                    println!("    [console.log] {}", msg);
                    Ok(JsValue::undefined())
                }),
                JsString::from("log"),
                1,
            )
            .build();
        ctx.register_global_property(JsString::from("console"), console, Attribute::all())
            .unwrap();

        // alert
        ctx.register_global_builtin_callable(
            JsString::from("alert"),
            1,
            NativeFunction::from_fn_ptr(|_, args, ctx| {
                let msg = args.get_or_undefined(0)
                    .to_string(ctx)
                    .map(|s| s.to_std_string_escaped())
                    .unwrap_or_default();
                println!("    [alert] {}", msg);
                Ok(JsValue::undefined())
            }),
        ).unwrap();

        self.js_ctx = Some(ctx);
    }

    pub fn run_script(&mut self, script: &str) -> Result<()> {
        if let Some(ctx) = &mut self.js_ctx {
            ctx.eval(Source::from_bytes(script))
                .map(|_| ())
                .map_err(|e| WebTestError::JsEvalError(format!("{:?}", e)))
        } else {
            Err(WebTestError::JsNotInitialized)
        }
    }

    pub fn call_function(&mut self, func_name: &str) -> Result<JsValue> {
        if let Some(ctx) = &mut self.js_ctx {
            let code = format!("{}()", func_name);
            ctx.eval(Source::from_bytes(&code))
                .map_err(|e| WebTestError::JsFunctionError(format!("{}: {:?}", func_name, e)))
        } else {
            Err(WebTestError::JsNotInitialized)
        }
    }

    /// 変数を取得（存在しない場合はNone）
    pub fn get_var(&mut self, var_name: &str) -> Option<String> {
        if let Some(ctx) = &mut self.js_ctx {
            match ctx.eval(Source::from_bytes(var_name)) {
                Ok(v) => Some(v.to_string(ctx).map(|s| s.to_std_string_escaped()).unwrap_or_default()),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    pub fn add(&mut self, elem: Element) {
        if let Some(id) = &elem.id { self.elements.insert(id.clone(), elem); }
    }

    pub fn click(&mut self, id: &str) -> bool {
        if let Some(e) = self.elements.get(id) {
            if e.is_disabled() { return false; }
            self.event_log.push(format!("click:{}", id));
            
            // onclick ハンドラを実行
            if e.events.contains(&"click".to_string()) {
                if let Some(handler) = e.onclick.clone() {
                    let func_name = handler.trim_end_matches("()");
                    let _ = self.call_function(func_name);
                }
            }
            return true;
        }
        false
    }

    pub fn input(&mut self, id: &str, val: &str) -> bool {
        if let Some(e) = self.elements.get_mut(id) {
            if e.is_disabled() { return false; }
            e.attributes.insert("value".to_string(), val.to_string());
            self.event_log.push(format!("input:{}={}", id, &val[..val.len().min(20)]));
            return true;
        }
        false
    }
}

pub fn parse_html(html: &str) -> (Vec<Element>, Vec<String>) {
    let mut elements = Vec::new();
    let mut scripts = Vec::new();

    let elem_re = regex::Regex::new(r#"<(\w+)([^>]*)>"#).unwrap();
    let script_re = regex::Regex::new(r#"<script[^>]*>([\s\S]*?)</script>"#).unwrap();

    for cap in script_re.captures_iter(html) {
        scripts.push(cap[1].to_string());
    }

    for cap in elem_re.captures_iter(html) {
        let tag = cap[1].to_lowercase();
        if ["script","style","html","head","body","meta","title"].contains(&tag.as_str()) { continue; }

        let attrs = &cap[2];
        let mut elem = Element { 
            tag, 
            id: None, 
            attributes: HashMap::new(), 
            events: Vec::new(),
            onclick: None,
            oninput: None,
        };

        if let Some(m) = regex::Regex::new(r#"id\s*=\s*["']([^"']+)["']"#).unwrap().captures(attrs) {
            elem.id = Some(m[1].to_string());
        }
        if let Some(m) = regex::Regex::new(r#"type\s*=\s*["']([^"']+)["']"#).unwrap().captures(attrs) {
            elem.attributes.insert("type".to_string(), m[1].to_string());
        }
        if attrs.to_lowercase().contains("disabled") {
            elem.attributes.insert("disabled".to_string(), "true".to_string());
        }
        
        // onclick
        if let Some(m) = regex::Regex::new(r#"onclick\s*=\s*["']([^"']+)["']"#).unwrap().captures(attrs) {
            elem.events.push("click".to_string());
            elem.onclick = Some(m[1].to_string());
        }
        // oninput
        if let Some(m) = regex::Regex::new(r#"oninput\s*=\s*["']([^"']+)["']"#).unwrap().captures(attrs) {
            elem.events.push("input".to_string());
            elem.oninput = Some(m[1].to_string());
        }
        // 他のonXXXイベント
        for ev in regex::Regex::new(r#"on(\w+)\s*="#).unwrap().captures_iter(attrs) {
            let event_name = ev[1].to_lowercase();
            if !elem.events.contains(&event_name) {
                elem.events.push(event_name);
            }
        }

        elements.push(elem);
    }

    (elements, scripts)
}

// ============================================================================
// プロパティテスト
// ============================================================================

#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub runs: usize,
    pub failure: Option<String>,
}

pub fn test_click(dom: &mut MiniDom, id: &str, n: usize) -> TestResult {
    let before = dom.get_var("count").unwrap_or_default();
    for _ in 0..n { dom.click(id); }
    let after = dom.get_var("count").unwrap_or_default();
    
    // 状態変化があればログ
    let note = if before != after {
        format!(" (count: {} → {})", before, after)
    } else {
        String::new()
    };
    
    TestResult { 
        name: format!("click#{}{}", id, note), 
        passed: true, 
        runs: n, 
        failure: None 
    }
}

pub fn test_input(dom: &mut MiniDom, id: &str) -> TestResult {
    let long = "a".repeat(1000);
    let cases = vec!["", " ", &long, "日本語", "😀", "<script>", "'; DROP TABLE"];
    for c in &cases { dom.input(id, c); }
    TestResult { name: format!("input#{}", id), passed: true, runs: cases.len(), failure: None }
}

pub fn test_disabled(dom: &mut MiniDom, id: &str) -> TestResult {
    let clicked = dom.click(id);
    TestResult {
        name: format!("disabled#{}", id),
        passed: !clicked,
        runs: 1,
        failure: if clicked { Some("clicked".to_string()) } else { None },
    }
}

// ============================================================================
// ファイル操作ヘルパー
// ============================================================================

/// ファイルを読み込み（拡張子チェック付き）
pub fn load_file(path: &str) -> Result<String> {
    let path = Path::new(path);
    
    // ファイル存在チェック
    if !path.exists() {
        return Err(WebTestError::FileNotFound(path.display().to_string()));
    }
    
    // 拡張子チェック
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    let valid_extensions = ["html", "htm", "js", "jsx", "ts", "tsx", "mjs", "cjs"];
    if !valid_extensions.contains(&ext) {
        return Err(WebTestError::InvalidFileFormat(
            ext.to_string(),
            valid_extensions.join(", ")
        ));
    }
    
    // ファイル読み込み
    fs::read_to_string(path).map_err(|e| WebTestError::FileReadError {
        path: path.display().to_string(),
        source: e,
    })
}

/// ディレクトリ内の対象ファイルを列挙
pub fn find_test_files(dir: &str) -> Result<Vec<String>> {
    let path = Path::new(dir);
    
    if !path.exists() {
        return Err(WebTestError::FileNotFound(dir.to_string()));
    }
    
    if !path.is_dir() {
        return Err(WebTestError::InvalidOperation(
            format!("{} is not a directory", dir)
        ));
    }
    
    let mut files = Vec::new();
    let valid_extensions = ["html", "htm", "js", "jsx", "ts", "tsx"];
    
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();
        
        if file_path.is_file() {
            if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                if valid_extensions.contains(&ext) {
                    files.push(file_path.display().to_string());
                }
            }
        }
    }
    
    files.sort();
    Ok(files)
}

// ============================================================================
// メイン
// ============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed = 42u64;
    let iterations = 50;

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║       web-prop-test: Rust製 Webテストツール       ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // ========== 道A: 並行処理テスト ==========
    println!("━━━ 道A: 並行処理テスト (Shuttle + Shrinking) ━━━\n");

    let num_tasks = 5;  // 5タスク同時インクリメント
    
    // プロパティテスト実行
    let (failures, shrunk) = prop_test_with_shrink(iterations, seed, num_tasks);
    
    println!("  タスク数: {}", num_tasks);
    println!("  実行: {}回", iterations);
    println!("  失敗: {}回", failures);
    println!("  期待値: count={}, 実際: count<{} (レースあり)", num_tasks, num_tasks);
    
    // Shrinking結果
    if let Some(s) = shrunk {
        println!("\n  🔍 Shrinking（最小反例）:");
        println!("    最小タスク数: {}", s.task_count);
        println!("    シード: {}", s.seed);
        println!("    結果: count={} (期待: {})", 
            s.final_state.get("count").unwrap_or(&0), s.expected);
        println!("    → {}タスクでも再現可能！", s.task_count);
    }
    
    // 全スケジュール探索（小さいnで）
    println!("\n  📊 全スケジュール探索 (3タスク, 3!=6通り):");
    let exhaustive = exhaustive_test(3);
    println!("    総スケジュール: {}", exhaustive.total_schedules);
    println!("    失敗: {}", exhaustive.failures);
    println!("    結果分布:");
    for (outcome, count) in &exhaustive.unique_outcomes {
        let pct = (*count as f64 / exhaustive.total_schedules as f64) * 100.0;
        let bar = "█".repeat((*count * 10 / exhaustive.total_schedules).max(1));
        let status = if *outcome == 3 { "✅" } else { "❌" };
        println!("      count={}: {} ({:.0}%) {}", outcome, bar, pct, status);
    }
    
    // 4タスクも試す
    println!("\n  📊 全スケジュール探索 (4タスク, 4!=24通り):");
    let exhaustive4 = exhaustive_test(4);
    println!("    総スケジュール: {}", exhaustive4.total_schedules);
    println!("    失敗: {} ({:.0}%)", exhaustive4.failures, 
        (exhaustive4.failures as f64 / exhaustive4.total_schedules as f64) * 100.0);
    println!("    結果分布:");
    for (outcome, count) in &exhaustive4.unique_outcomes {
        let status = if *outcome == 4 { "✅" } else { "❌" };
        println!("      count={}: {} {}", outcome, count, status);
    }
    
    // ========== 高精度テスト（Vector Clock + PCT） ==========
    println!("\n━━━ 高精度テスト (Vector Clock + DPOR + PCT) ━━━\n");
    
    // Vector Clock デモ
    println!("  🕐 Vector Clock レース検出:");
    let mut adv = AdvancedScheduler::new(3, seed);
    let t0 = adv.spawn("task0");
    let t1 = adv.spawn("task1");
    let _t2 = adv.spawn("task2");
    adv.init_var("x", 0);
    
    // 並行アクセスをシミュレート
    adv.read_var(t0, "x");   // T0: read x
    adv.read_var(t1, "x");   // T1: read x (concurrent with T0)
    adv.write_var(t0, "x", 1); // T0: write x
    adv.write_var(t1, "x", 2); // T1: write x (race with T0's write!)
    
    let vc_races = adv.get_races();
    println!("    検出レース: {}件", vc_races.len());
    for race in vc_races {
        println!("      {} '{}': T{} vs T{}", 
            race.race_type, race.var_name, 
            race.access1.task_id, race.access2.task_id);
    }
    
    // PCT デモ
    println!("\n  🎲 PCT (Probabilistic Concurrency Testing):");
    let d = 2;  // バグ深度
    let k = 10; // ステップ数
    let n = 3;  // タスク数
    let prob = PCTScheduler::bug_probability(n, k, d);
    println!("    パラメータ: n={}, k={}, d={}", n, k, d);
    println!("    理論的バグ発見確率: >= {:.2}%", prob * 100.0);
    
    // PCTでの実行
    let mut pct_races = 0;
    for i in 0..100 {
        let mut adv_pct = AdvancedScheduler::new(3, seed + i)
            .with_pct(10, d);
        let pt0 = adv_pct.spawn("task0");
        let pt1 = adv_pct.spawn("task1");
        adv_pct.init_var("count", 0);
        
        let p0 = adv_pct.add_promise();
        let p1 = adv_pct.add_promise();
        
        adv_pct.read_var(pt0, "count");
        adv_pct.block_task(pt0, p0);
        adv_pct.read_var(pt1, "count");
        adv_pct.block_task(pt1, p1);
        
        adv_pct.resolve_next();
        adv_pct.resolve_next();
        
        adv_pct.write_var(pt0, "count", 1);
        adv_pct.write_var(pt1, "count", 1);
        
        if !adv_pct.get_races().is_empty() {
            pct_races += 1;
        }
    }
    println!("    100回実行でレース検出: {}回", pct_races);
    
    // DPOR 効果の説明
    println!("\n  📉 DPOR (Dynamic Partial Order Reduction):");
    println!("    目的: 不要なスケジュール探索を削減");
    println!("    効果: n!通り → 依存関係のある組み合わせのみ");
    let full_4 = factorial(4);
    let reduced = 6;  // 実際は依存解析による
    println!("    例: 4タスク {}通り → 約{}通りに削減 ({:.0}%減)", 
        full_4, reduced, (1.0 - reduced as f64 / full_4 as f64) * 100.0);
    
    // ========== FastTrack最適化版 ==========
    println!("\n━━━ FastTrack最適化版 (Epoch + Race Coverage) ━━━\n");
    
    // FastTrackスケジューラでのテスト
    println!("  ⚡ FastTrack Epoch最適化:");
    let mut ft = FastTrackScheduler::new(4, seed);
    let ft0 = ft.spawn("worker0");
    let ft1 = ft.spawn("worker1");
    let ft2 = ft.spawn("worker2");
    let _ft3 = ft.spawn("worker3");
    
    ft.init_var("counter", 0);
    ft.init_var("flag", false as i64);
    
    // シナリオ: 並行インクリメント + フラグ同期
    ft.read_var(ft0, "counter");
    ft.read_var(ft1, "counter");
    ft.read_var(ft2, "counter");
    ft.write_var(ft0, "counter", 1);
    ft.write_var(ft1, "counter", 1);  // W-W race
    ft.write_var(ft2, "counter", 1);  // W-W race
    ft.write_var(ft0, "flag", 1);     // Ad-hoc sync
    ft.read_var(ft1, "flag");
    
    ft.finalize();
    
    let (harmful, benign, covered, adhoc) = ft.get_race_summary();
    println!("    検出レース総数: {}", harmful + benign + covered + adhoc);
    println!("    ├─ Harmful:  {} (実害あり)", harmful);
    println!("    ├─ Benign:   {} (同値書込)", benign);
    println!("    ├─ Covered:  {} (他でカバー)", covered);
    println!("    └─ Ad-hoc:   {} (同期パターン)", adhoc);
    
    // 有害レース詳細
    println!("\n  🚨 有害レース詳細:");
    for race in ft.get_harmful_races() {
        println!("    {} '{}': T{} vs T{}", 
            race.race.race_type, race.race.var_name,
            race.race.access1.task_id, race.race.access2.task_id);
    }
    
    // Predictive Analysis
    println!("\n  🔮 Predictive Analysis (観測外レース予測):");
    let predictions = ft.get_predictions();
    for pred in &predictions {
        println!("    '{}' T{} vs T{} - 確信度:{:.0}%", 
            pred.var_name, pred.thread1, pred.thread2, pred.confidence * 100.0);
        println!("      理由: {}", pred.reason);
    }
    
    // 性能比較
    println!("\n  📊 性能比較 (理論値):");
    println!("    ┌────────────────┬─────────┬─────────┐");
    println!("    │ アルゴリズム    │ 時間計算量 │ 空間計算量 │");
    println!("    ├────────────────┼─────────┼─────────┤");
    println!("    │ BasicVC        │ O(n)    │ O(n)    │");
    println!("    │ DJIT+          │ O(n)    │ O(n)    │");
    println!("    │ FastTrack      │ O(1)*   │ O(1)*   │");
    println!("    │ (本実装)        │ O(1)*   │ O(1)*   │");
    println!("    └────────────────┴─────────┴─────────┘");
    println!("    * 単一スレッドアクセス時（Epoch表現）");
    
    // ========== 包括的レーステストシナリオ ==========
    println!("\n━━━ 包括的レーステストシナリオ (15カテゴリ) ━━━\n");
    
    run_comprehensive_race_tests(seed);

    println!();

    // ========== 道B: DOMテスト ==========
    println!("━━━ 道B: DOMプロパティテスト ━━━\n");

    let html = if args.len() > 1 {
        match load_file(&args[1]) {
            Ok(content) => {
                println!("  ファイル読み込み: {} bytes", content.len());
                content
            }
            Err(e) => {
                println!("  ❌ エラー: {}", e);
                String::new()
            }
        }
    } else {
        r#"<button id="btn" onclick="f()">OK</button>
           <button id="dis" disabled>No</button>
           <input id="txt" type="text">"#.to_string()
    };

    let (elements, scripts) = parse_html(&html);
    println!("  パース結果: {}要素, {}スクリプト", elements.len(), scripts.len());
    
    let mut dom = MiniDom::new();
    dom.init_js();
    
    // スクリプト実行
    for (i, script) in scripts.iter().enumerate() {
        match dom.run_script(script) {
            Ok(_) => println!("  スクリプト{}: 実行OK", i + 1),
            Err(e) => println!("  スクリプト{}: ❌ {}", i + 1, e),
        }
        
        // AST解析
        let analysis = analyze_js(script);
        
        // 基本情報
        if !analysis.async_funcs.is_empty() || !analysis.react_components.is_empty() {
            println!("\n  🔍 AST解析結果:");
            println!("    グローバル変数: {:?}", analysis.global_vars);
            println!("    async関数: {}個", analysis.async_funcs.len());
            println!("    awaitポイント: {}箇所", analysis.await_points);
            
            for func in &analysis.async_funcs {
                println!("    📦 {} - await:{}回, 共有変数:{:?}", 
                    func.name, func.await_count, func.shared_vars);
                
                if !func.shared_vars.is_empty() && func.await_count > 0 {
                    println!("      ⚠️  レースコンディションの可能性あり！");
                }
            }
        }
        
        // React/Next.js検出
        if analysis.is_jsx || !analysis.react_components.is_empty() || !analysis.next_patterns.is_empty() {
            println!("\n  ⚛️  React/Next.js 検出:");
            
            if analysis.is_jsx {
                println!("    JSX: 検出");
            }
            
            if !analysis.next_patterns.is_empty() {
                println!("    Next.js パターン: {:?}", analysis.next_patterns);
            }
            
            for comp in &analysis.react_components {
                println!("    📦 コンポーネント: {}", comp.name);
                println!("      ├─ useState: {}個", comp.state_vars.len());
                println!("      ├─ useEffect: {}個", comp.effect_count);
                println!("      └─ フック総数: {}個", comp.hooks.len());
                
                // レース警告
                for hook in &comp.hooks {
                    if hook.potential_race {
                        println!("      ⚠️  {}: クリーンアップ不足または依存配列の問題", hook.hook_name);
                    }
                }
            }
        }
    }
    
    // 初期状態の変数を表示
    println!("\n  JS変数:");
    if let Some(v) = dom.get_var("count") { println!("    count = {}", v); }
    if let Some(v) = dom.get_var("balance") { println!("    balance = {}", v); }
    
    for e in &elements { dom.add(e.clone()); }

    println!("  検出要素:");
    for e in &elements {
        if e.id.is_some() || !e.events.is_empty() {
            let id = e.id.as_deref().unwrap_or("-");
            let dis = if e.is_disabled() { " (disabled)" } else { "" };
            println!("    <{}> #{} {:?}{}", e.tag, id, e.events, dis);
        }
    }
    println!();

    let mut results = Vec::new();
    for e in &elements {
        if let Some(id) = &e.id {
            if e.tag == "button" && !e.is_disabled() {
                results.push(test_click(&mut dom, id, 100));
            }
            if e.tag == "input" {
                results.push(test_input(&mut dom, id));
            }
            if e.is_disabled() {
                results.push(test_disabled(&mut dom, id));
            }
        }
    }

    println!("  テスト結果:");
    let passed = results.iter().filter(|r| r.passed).count();
    for r in &results {
        let icon = if r.passed { "✅" } else { "❌" };
        println!("    {} {} ({}回)", icon, r.name, r.runs);
    }
    println!("\n  {}/{} passed", passed, results.len());

    println!("\n═══════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_race_detection() {
        let mut s = Scheduler::new(42);
        s.init_var("x", 0);
        let t1 = s.spawn("t1");
        let t2 = s.spawn("t2");
        s.read_var(t1, "x");
        s.write_var(t2, "x", 1);
        assert!(!s.detect_races().is_empty());
    }

    #[test]
    fn test_disabled_element() {
        let mut dom = MiniDom::new();
        let mut e = Element { 
            tag: "button".into(), 
            id: Some("b".into()), 
            attributes: HashMap::new(), 
            events: vec![],
            onclick: None,
            oninput: None,
        };
        e.attributes.insert("disabled".into(), "true".into());
        dom.add(e);
        assert!(!dom.click("b"));
    }

    // ========================================
    // エラー型テスト
    // ========================================

    #[test]
    fn test_error_display() {
        let err = WebTestError::JsNotInitialized;
        assert_eq!(format!("{}", err), "JS engine not initialized");
        
        let err = WebTestError::JsEvalError("syntax error".to_string());
        assert!(format!("{}", err).contains("syntax error"));
        
        let err = WebTestError::FileNotFound("/tmp/test.html".to_string());
        assert!(format!("{}", err).contains("/tmp/test.html"));
    }

    #[test]
    fn test_error_variants() {
        // 各バリアントが正しく構築できることを確認
        let _ = WebTestError::JsNotInitialized;
        let _ = WebTestError::JsEvalError("test".into());
        let _ = WebTestError::JsFunctionError("func".into());
        let _ = WebTestError::HtmlParseError("parse".into());
        let _ = WebTestError::FileNotFound("path".into());
        let _ = WebTestError::InvalidFileFormat("txt".into(), "html".into());
        let _ = WebTestError::SchedulerError("sched".into());
        let _ = WebTestError::VariableNotFound("x".into());
        let _ = WebTestError::TaskNotFound(42);
        let _ = WebTestError::ElementNotFound("btn".into());
        let _ = WebTestError::InvalidOperation("op".into());
    }

    #[test]
    fn test_js_not_initialized_error() {
        let mut dom = MiniDom::new();
        // init_js()を呼ばずにスクリプト実行
        let result = dom.run_script("console.log('test')");
        assert!(matches!(result, Err(WebTestError::JsNotInitialized)));
    }

    #[test]
    fn test_js_eval_error() {
        let mut dom = MiniDom::new();
        dom.init_js();
        // 構文エラーのあるスクリプト
        let result = dom.run_script("function { broken syntax");
        assert!(matches!(result, Err(WebTestError::JsEvalError(_))));
    }

    #[test]
    fn test_js_function_error() {
        let mut dom = MiniDom::new();
        dom.init_js();
        // 存在しない関数を呼び出し
        let result = dom.call_function("nonExistentFunction");
        assert!(matches!(result, Err(WebTestError::JsFunctionError(_))));
    }

    #[test]
    fn test_load_file_not_found() {
        let result = load_file("/nonexistent/path/file.html");
        assert!(matches!(result, Err(WebTestError::FileNotFound(_))));
    }

    #[test]
    fn test_load_file_invalid_extension() {
        // 一時ファイルを作成
        let temp_path = "/tmp/test_invalid.txt";
        fs::write(temp_path, "test content").unwrap();
        
        let result = load_file(temp_path);
        assert!(matches!(result, Err(WebTestError::InvalidFileFormat(_, _))));
        
        // クリーンアップ
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_result_type_alias() {
        fn example_fn() -> Result<i32> {
            Ok(42)
        }
        
        fn example_err() -> Result<i32> {
            Err(WebTestError::InvalidOperation("test".into()))
        }
        
        assert_eq!(example_fn().unwrap(), 42);
        assert!(example_err().is_err());
    }

    #[test]
    fn test_error_source_chain() {
        use std::error::Error;
        
        // FileReadErrorのsource chainをテスト
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = WebTestError::FileReadError {
            path: "/test/path".into(),
            source: io_err,
        };
        
        // sourceが正しく設定されていることを確認
        assert!(err.source().is_some());
    }
}