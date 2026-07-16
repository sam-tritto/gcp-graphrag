import html
from pyvis.network import Network

def generate_bbn_pyvis_graph(
    graph_records,
    user_stats,
    inference=None,
    bbn_model=None,
    show_subconcepts=True,
    domain_filter=None,
    rankdir="LR",
    output_filename="data/mastery_graph.html"
):
    """
    Constructs an interactive Pyvis-based Vis.js network representing the Domain-Service-SubConcept hierarchy,
    along with all connected Use Cases, Functional Roles, CLI Commands, Hierarchy Nodes, and Anti-patterns.
    Displays all nodes to create a rich constellation layout, highlighting nodes of the selected domain
    while fading out the rest.
    """
    # Create pyvis network with light theme background (#f8f9fa).
    net = Network(height="500px", width="100%", bgcolor="#f8f9fa", font_color="#202124", directed=True)
    
    # Track nodes and edges to avoid duplicates
    rendered_nodes = set()
    rendered_edges = set()
    
    # Map each node ID to the domains it belongs to
    node_domains = {}
    for r in graph_records:
        d_name = r.get("domain")
        s_name = r.get("service")
        sub_name = r.get("subconcept")
        label_out = r.get("label_out")
        val_out = r.get("val_out")
        label_in = r.get("label_in")
        val_in = r.get("val_in")
        pref_service = r.get("pref_service")
        ap_desc = r.get("ap_desc")
        ap_res = r.get("ap_res")
        
        if d_name:
            d_node = f"{d_name}_Domain"
            node_domains.setdefault(d_node, set()).add(d_name)
            
            if s_name:
                s_node = f"{s_name}_Service"
                node_domains.setdefault(s_node, set()).add(d_name)
                
                if sub_name:
                    sub_node = f"{sub_name}_SubConcept"
                    node_domains.setdefault(sub_node, set()).add(d_name)
                    
                if val_out and label_out:
                    leaf_node = f"{val_out}_{label_out}"
                    node_domains.setdefault(leaf_node, set()).add(d_name)
                    
                if val_in and label_in:
                    leaf_node = f"{val_in}_{label_in}"
                    node_domains.setdefault(leaf_node, set()).add(d_name)
                    
                if pref_service:
                    pref_node = f"{pref_service}_Service"
                    node_domains.setdefault(pref_node, set()).add(d_name)
                    
                if ap_desc:
                    ap_node = f"{ap_desc}_AntiPattern"
                    node_domains.setdefault(ap_node, set()).add(d_name)
                    if ap_res:
                        res_node = f"{ap_res}_Service"
                        node_domains.setdefault(res_node, set()).add(d_name)

    def check_active(node_id):
        if not domain_filter or domain_filter == "All Domains":
            return True
        return node_id in node_domains and domain_filter in node_domains[node_id]

    for r in graph_records:
        d_name = r.get("domain")
        if not d_name:
            continue
            
        d_node = f"{d_name}_Domain"
        
        # 1. Domain Node
        p_domain = 0.5
        if inference and bbn_model and d_node in bbn_model.nodes():
            try:
                p_domain = float(inference.query(variables=[d_node], show_progress=False).values[1])
            except Exception:
                d_stats = user_stats.get("domains", {}).get(d_name, {"alpha": 1, "beta": 1})
                p_domain = d_stats["beta"] / (d_stats["alpha"] + d_stats["beta"])
        else:
            d_stats = user_stats.get("domains", {}).get(d_name, {"alpha": 1, "beta": 1})
            p_domain = d_stats["beta"] / (d_stats["alpha"] + d_stats["beta"])
            
        is_d_active = check_active(d_node)
        
        if is_d_active:
            d_fill = "#e8f0fe"
            d_font_color = "#1a73e8"
            d_border_width = 3
            d_size = 24
            if p_domain >= 0.7:
                d_border = "#1e8e3e"  # Green
                status_text = "Mastered"
                status_color = "#137333"
            elif p_domain >= 0.3:
                d_border = "#f9ab00"  # Yellow
                status_text = "Reviewing"
                status_color = "#b06000"
            else:
                d_border = "#d93025"  # Red
                status_text = "Struggling"
                status_color = "#c5221f"
        else:
            d_fill = "rgba(241, 243, 244, 0.15)"
            d_border = "rgba(218, 220, 224, 0.15)"
            d_font_color = "rgba(128, 128, 128, 0.15)"
            d_border_width = 1
            d_size = 20
            status_text = "Faded"
            status_color = "#9aa0a6"
            
        # Use single space to bypass Pyvis fallback to ID label
        d_label = " "
        
        d_tooltip = f"""
        <div style="font-family: 'Outfit', sans-serif; color: #202124;">
            <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #1a73e8; font-weight: 700;">Active Certification Domain</div>
            <div style="font-size: 15px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(d_name)}</div>
            <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
            <div style="font-size: 12px; color: #5f6368; line-height: 1.5; font-weight: 400;">
                This syllabus domain represents a fundamental pillar of your targeted Google Cloud certification. 
                Your real-time mastery probability is estimated dynamically through Bayesian updates based on active quiz performance.
            </div>
            <div style="margin-top: 10px; display: flex; align-items: center; justify-content: space-between; font-size: 11px;">
                <span style="color: #5f6368;">Target Mastery: 70%</span>
                <span>Current Status: <strong style="color: {status_color};">{status_text} ({p_domain:.1%})</strong></span>
            </div>
        </div>
        """
        
        if d_node not in rendered_nodes:
            net.add_node(
                d_node,
                label=d_label,
                title=d_tooltip,
                color={"background": d_fill, "border": d_border if is_d_active else d_border, "highlight": {"background": d_fill, "border": "#1a73e8"}},
                font={"color": d_font_color, "size": 10, "face": "Outfit"},
                shape="dot",
                size=d_size,
                borderWidth=d_border_width,
                level=0
            )
            rendered_nodes.add(d_node)
            
        # 2. Service Node
        s_name = r.get("service")
        if not s_name:
            continue
            
        s_node = f"{s_name}_Service"
        
        p_service = 0.5
        if inference and bbn_model and s_node in bbn_model.nodes():
            try:
                p_service = float(inference.query(variables=[s_node], show_progress=False).values[1])
            except Exception:
                s_stats = user_stats.get("services", {}).get(s_name, {"alpha": 1, "beta": 1})
                p_service = s_stats["beta"] / (s_stats["alpha"] + s_stats["beta"])
        else:
            s_stats = user_stats.get("services", {}).get(s_name, {"alpha": 1, "beta": 1})
            p_service = s_stats["beta"] / (s_stats["alpha"] + s_stats["beta"])
            
        is_s_active = check_active(s_node)
        
        if is_s_active:
            s_fill = "#e6f4ea"
            s_font_color = "#202124"
            s_border_width = 2.5
            s_size = 16
            if p_service >= 0.7:
                s_border = "#1e8e3e"
                s_status = "Mastered"
                s_status_color = "#137333"
            elif p_service >= 0.3:
                s_border = "#f9ab00"
                s_status = "Reviewing"
                s_status_color = "#b06000"
            else:
                s_border = "#d93025"
                s_status = "Struggling"
                s_status_color = "#c5221f"
        else:
            s_fill = "rgba(241, 243, 244, 0.15)"
            s_border = "rgba(218, 220, 224, 0.15)"
            s_font_color = "rgba(128, 128, 128, 0.15)"
            s_border_width = 1
            s_size = 12
            s_status = "Faded"
            s_status_color = "#9aa0a6"
            
        # Use single space to bypass Pyvis fallback to ID label
        s_label = " "
        
        s_tooltip = f"""
        <div style="font-family: 'Outfit', sans-serif; color: #202124;">
            <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #137333; font-weight: 700;">GCP System Service</div>
            <div style="font-size: 15px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(s_name)}</div>
            <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
            <div style="font-size: 12px; color: #5f6368; line-height: 1.5; font-weight: 400;">
                A key Google Cloud product or system component associated with this domain. 
                Its status reflects aggregate user data, influencing Thompson sampling choices for active practice questions.
            </div>
            <div style="margin-top: 10px; display: flex; align-items: center; justify-content: space-between; font-size: 11px;">
                <span style="color: #5f6368;">Intermediate Node</span>
                <span>Status: <strong style="color: {s_status_color};">{s_status} ({p_service:.1%})</strong></span>
            </div>
        </div>
        """
        
        if s_node not in rendered_nodes:
            net.add_node(
                s_node,
                label=s_label,
                title=s_tooltip,
                color={"background": s_fill, "border": s_border if is_s_active else s_border, "highlight": {"background": s_fill, "border": "#1a73e8"}},
                font={"color": s_font_color, "size": 9, "face": "Outfit"},
                shape="dot",
                size=s_size,
                borderWidth=s_border_width,
                level=1
            )
            rendered_nodes.add(s_node)
            
        d_s_edge = (d_node, s_node)
        if d_s_edge not in rendered_edges:
            edge_color = "#bdc1c6" if (is_d_active and is_s_active) else "rgba(218, 220, 224, 0.12)"
            net.add_edge(d_node, s_node, color={"color": edge_color, "highlight": "#1a73e8"})
            rendered_edges.add(d_s_edge)
            
        # 3. Subconcept Node
        sub_name = r.get("subconcept")
        if show_subconcepts and sub_name:
            sub_node = f"{sub_name}_SubConcept"
            
            p_sub = 0.5
            if inference and bbn_model and sub_node in bbn_model.nodes():
                try:
                    p_sub = float(inference.query(variables=[sub_node], show_progress=False).values[1])
                except Exception:
                    sub_stats = user_stats.get("subconcepts", {}).get(sub_name, {"alpha": 1, "beta": 1})
                    p_sub = sub_stats["beta"] / (sub_stats["alpha"] + sub_stats["beta"])
            else:
                sub_stats = user_stats.get("subconcepts", {}).get(sub_name, {"alpha": 1, "beta": 1})
                p_sub = sub_stats["beta"] / (sub_stats["alpha"] + sub_stats["beta"])
                
            is_sub_active = check_active(sub_node)
            
            if is_sub_active:
                sub_fill = "#fef7e0"
                sub_border_width = 2
                sub_size = 10
                if p_sub >= 0.7:
                    sub_border = "#1e8e3e"
                    sub_status = "Mastered"
                    sub_status_color = "#137333"
                elif p_sub >= 0.3:
                    sub_border = "#f9ab00"
                    sub_status = "Reviewing"
                    sub_status_color = "#b06000"
                else:
                    sub_border = "#d93025"
                    sub_status = "Struggling"
                    sub_status_color = "#c5221f"
            else:
                sub_fill = "rgba(241, 243, 244, 0.15)"
                sub_border = "rgba(218, 220, 224, 0.15)"
                sub_border_width = 1
                sub_size = 8
                sub_status = "Faded"
                sub_status_color = "#9aa0a6"
                
            # Use single space to bypass Pyvis fallback to ID label
            sub_label = " "
            
            sub_tooltip = f"""
            <div style="font-family: 'Outfit', sans-serif; color: #202124;">
                <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #b06000; font-weight: 700;">Architectural Subconcept</div>
                <div style="font-size: 15px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(sub_name)}</div>
                <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
                <div style="font-size: 12px; color: #5f6368; line-height: 1.5; font-weight: 400;">
                    A granular configuration detail, service integration pattern, or common troubleshooting pathway. 
                    Interactive practice quiz submissions update this node's prior or posterior probability state directly.
                </div>
                <div style="margin-top: 10px; display: flex; align-items: center; justify-content: space-between; font-size: 11px;">
                    <span style="color: #5f6368;">Leaf Node</span>
                    <span>Status: <strong style="color: {sub_status_color};">{sub_status} ({p_sub:.1%})</strong></span>
                </div>
            </div>
            """
            
            if sub_node not in rendered_nodes:
                net.add_node(
                    sub_node,
                    label=sub_label,
                    title=sub_tooltip,
                    color={"background": sub_fill, "border": sub_border if is_sub_active else sub_border, "highlight": {"background": sub_fill, "border": "#1a73e8"}},
                    font={"color": "#202124", "size": 8, "face": "Outfit"},
                    shape="dot",
                    size=sub_size,
                    borderWidth=sub_border_width,
                    level=2
                )
                rendered_nodes.add(sub_node)
                
            s_sub_edge = (s_node, sub_node)
            if s_sub_edge not in rendered_edges:
                edge_color = "#dadce0" if (is_s_active and is_sub_active) else "rgba(218, 220, 224, 0.12)"
                net.add_edge(s_node, sub_node, color={"color": edge_color, "highlight": "#1a73e8"})
                rendered_edges.add(s_sub_edge)

        # 4. Outgoing Third-Tier Leaf Node
        rel_out = r.get("rel_out")
        label_out = r.get("label_out")
        val_out = r.get("val_out")
        
        if rel_out and label_out and val_out:
            leaf_node = f"{val_out}_{label_out}"
            is_leaf_active = check_active(leaf_node)
            
            # Map tag styles
            if is_leaf_active:
                if label_out == "UseCase":
                    leaf_fill = "#e3f2fd" # Soft Blue
                    label_color = "#1a73e8"
                elif label_out == "CLICommand":
                    leaf_fill = "#f3e5f5" # Soft Purple
                    label_color = "#9c27b0"
                elif label_out == "FunctionalRole":
                    leaf_fill = "#e0f2f1" # Soft Teal
                    label_color = "#009688"
                elif label_out == "HierarchyNode":
                    leaf_fill = "#efebe9" # Soft Brown
                    label_color = "#795548"
                else:
                    leaf_fill = "#f5f5f5" # Soft Grey
                    label_color = "#757575"
                leaf_border = "#bdc1c6"
                leaf_border_width = 1.5
                leaf_size = 8
            else:
                leaf_fill = "rgba(241, 243, 244, 0.15)"
                leaf_border = "rgba(218, 220, 224, 0.15)"
                label_color = "rgba(128, 128, 128, 0.15)"
                leaf_border_width = 1
                leaf_size = 6
                
            leaf_tooltip = f"""
            <div style="font-family: 'Outfit', sans-serif; color: #202124;">
                <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: {label_color}; font-weight: 700;">{label_out} (GCP Database Relation)</div>
                <div style="font-size: 14px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(val_out)}</div>
                <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
                <div style="font-size: 11px; color: #5f6368; line-height: 1.4;">
                    Active GCP structural element mapped in the Neo4j database graph for this target certification.
                </div>
            </div>
            """
            
            if leaf_node not in rendered_nodes:
                net.add_node(
                    leaf_node,
                    label=" ",
                    title=leaf_tooltip,
                    color={"background": leaf_fill, "border": leaf_border, "highlight": {"background": leaf_fill, "border": "#1a73e8"}},
                    font={"color": "#5f6368", "size": 8, "face": "Outfit"},
                    shape="dot",
                    size=leaf_size,
                    borderWidth=leaf_border_width,
                    level=3
                )
                rendered_nodes.add(leaf_node)
                
            s_leaf_edge = (s_node, leaf_node)
            if s_leaf_edge not in rendered_edges:
                edge_color = "#e0e0e0" if (is_s_active and is_leaf_active) else "rgba(218, 220, 224, 0.12)"
                net.add_edge(s_node, leaf_node, title=rel_out, label=rel_out, color={"color": edge_color, "highlight": "#1a73e8"}, font={"size": 7, "color": "rgba(154, 160, 166, 0.5)"})
                rendered_edges.add(s_leaf_edge)

        # 5. Incoming Third-Tier Leaf Node
        rel_in = r.get("rel_in")
        label_in = r.get("label_in")
        val_in = r.get("val_in")
        
        if rel_in and label_in and val_in:
            leaf_node = f"{val_in}_{label_in}"
            is_leaf_active = check_active(leaf_node)
            
            if is_leaf_active:
                if label_in == "UseCase":
                    leaf_fill = "#e3f2fd"
                    label_color = "#1a73e8"
                elif label_in == "CLICommand":
                    leaf_fill = "#f3e5f5"
                    label_color = "#9c27b0"
                elif label_in == "FunctionalRole":
                    leaf_fill = "#e0f2f1"
                    label_color = "#009688"
                elif label_in == "HierarchyNode":
                    leaf_fill = "#efebe9"
                    label_color = "#795548"
                else:
                    leaf_fill = "#f5f5f5"
                    label_color = "#757575"
                leaf_border = "#bdc1c6"
                leaf_border_width = 1.5
                leaf_size = 8
            else:
                leaf_fill = "rgba(241, 243, 244, 0.15)"
                leaf_border = "rgba(218, 220, 224, 0.15)"
                label_color = "rgba(128, 128, 128, 0.15)"
                leaf_border_width = 1
                leaf_size = 6
                
            leaf_tooltip = f"""
            <div style="font-family: 'Outfit', sans-serif; color: #202124;">
                <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: {label_color}; font-weight: 700;">{label_in} (GCP Database Relation)</div>
                <div style="font-size: 14px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(val_in)}</div>
                <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
                <div style="font-size: 11px; color: #5f6368; line-height: 1.4;">
                    Active GCP structural element mapped in the Neo4j database graph for this target certification.
                </div>
            </div>
            """
            
            if leaf_node not in rendered_nodes:
                net.add_node(
                    leaf_node,
                    label=" ",
                    title=leaf_tooltip,
                    color={"background": leaf_fill, "border": leaf_border, "highlight": {"background": leaf_fill, "border": "#1a73e8"}},
                    font={"color": "#5f6368", "size": 8, "face": "Outfit"},
                    shape="dot",
                    size=leaf_size,
                    borderWidth=leaf_border_width,
                    level=3
                )
                rendered_nodes.add(leaf_node)
                
            leaf_s_edge = (leaf_node, s_node)
            if leaf_s_edge not in rendered_edges:
                edge_color = "#e0e0e0" if (is_leaf_active and is_s_active) else "rgba(218, 220, 224, 0.12)"
                net.add_edge(leaf_node, s_node, title=rel_in, label=rel_in, color={"color": edge_color, "highlight": "#1a73e8"}, font={"size": 7, "color": "rgba(154, 160, 166, 0.5)"})
                rendered_edges.add(leaf_s_edge)

        # 6. Preferable Over Decision Boundary
        pref_service = r.get("pref_service")
        if pref_service:
            pref_node = f"{pref_service}_Service"
            if pref_node in rendered_nodes:
                pref_edge = (s_node, pref_node)
                if pref_edge not in rendered_edges:
                    edge_color = "#ea4335" if (is_s_active and check_active(pref_node)) else "rgba(218, 220, 224, 0.12)"
                    net.add_edge(s_node, pref_node, title="PREFERABLE_OVER", label="PREFERABLE_OVER", color={"color": edge_color, "highlight": "#d93025"}, width=2, dashes=True, font={"size": 8, "color": "#d93025"})
                    rendered_edges.add(pref_edge)

        # 7. AntiPattern & Pitfall Nodes
        ap_desc = r.get("ap_desc")
        ap_res = r.get("ap_res")
        if ap_desc:
            ap_node = f"{ap_desc}_AntiPattern"
            is_ap_active = check_active(ap_node)
            
            if is_ap_active:
                ap_fill = "#fce8e6"
                ap_border = "#d93025"
                ap_font_color = "#c5221f"
                ap_border_width = 2
                ap_size = 12
            else:
                ap_fill = "rgba(241, 243, 244, 0.15)"
                ap_border = "rgba(218, 220, 224, 0.15)"
                ap_font_color = "rgba(128, 128, 128, 0.15)"
                ap_border_width = 1
                ap_size = 8
            
            ap_tooltip = f"""
            <div style="font-family: 'Outfit', sans-serif; color: #202124;">
                <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #c5221f; font-weight: 700;">GCP AntiPattern & Pitfall</div>
                <div style="font-size: 14px; font-weight: 600; margin-top: 4px; color: #202124; line-height: 1.2;">{html.escape(ap_desc)}</div>
                <div style="height: 1px; background-color: #dadce0; margin: 8px 0;"></div>
                <div style="font-size: 11px; color: #5f6368; line-height: 1.4;">
                    Warning: Avoid this common design pitfall in practice or exam scenarios.
                </div>
                {f'<div style="margin-top: 8px; font-size: 11px; font-weight: 600; color: #137333;">Resolution: Resolved by {html.escape(ap_res)}</div>' if ap_res else ''}
            </div>
            """
            
            if ap_node not in rendered_nodes:
                net.add_node(
                    ap_node,
                    label=" ",
                    title=ap_tooltip,
                    color={"background": ap_fill, "border": ap_border, "highlight": {"background": ap_fill, "border": "#d93025"}},
                    font={"color": ap_font_color, "size": 8, "face": "Outfit"},
                    shape="dot",
                    size=ap_size,
                    borderWidth=ap_border_width,
                    level=3
                )
                rendered_nodes.add(ap_node)
                
            s_ap_edge = (s_node, ap_node)
            if s_ap_edge not in rendered_edges:
                edge_color = "#d93025" if (is_s_active and is_ap_active) else "rgba(218, 220, 224, 0.12)"
                net.add_edge(s_node, ap_node, title="COMMON_PITFALL", label="COMMON_PITFALL", color={"color": edge_color, "highlight": "#d93025"}, width=1.5, dashes=True, font={"size": 7, "color": "#c5221f"})
                rendered_edges.add(s_ap_edge)
                
            if ap_res:
                res_node = f"{ap_res}_Service"
                if res_node in rendered_nodes:
                    ap_res_edge = (ap_node, res_node)
                    if ap_res_edge not in rendered_edges:
                        edge_color = "#1e8e3e" if (is_ap_active and check_active(res_node)) else "rgba(218, 220, 224, 0.12)"
                        net.add_edge(ap_node, res_node, title="RESOLVED_BY", label="RESOLVED_BY", color={"color": edge_color, "highlight": "#1e8e3e"}, width=1.5, font={"size": 7, "color": "#137333"})
                        rendered_edges.add(ap_res_edge)
                        
    # Setup graph physics and visual styling options
    hierarchical_options = ""
    if rankdir in ["TB", "LR"]:
        direction = "UD" if rankdir == "TB" else "LR"
        node_spacing = 90 if rankdir == "TB" else 100
        level_sep = 160 if rankdir == "TB" else 150
        hierarchical_options = f"""
        "hierarchical": {{
            "enabled": true,
            "direction": "{direction}",
            "sortMethod": "directed",
            "nodeSpacing": {node_spacing},
            "levelSeparation": {level_sep}
        }},
        """

    options_json = f"""
    var options = {{
      "nodes": {{
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shadow": {{
          "enabled": true,
          "color": "rgba(0, 0, 0, 0.04)",
          "size": 4,
          "x": 1,
          "y": 1
        }}
      }},
      "edges": {{
        "width": 1.5,
        "arrows": {{
          "to": {{
            "enabled": true,
            "scaleFactor": 0.4
          }}
        }},
        "smooth": {{
          "type": "cubicBezier",
          "forceDirection": "none",
          "roundness": 0.4
        }}
      }},
      "layout": {{
        {hierarchical_options}
        "randomSeed": 42
      }},
      "physics": {{
        "enabled": {str(rankdir not in ["TB", "LR"]).lower()},
        "forceAtlas2Based": {{
          "gravitationalConstant": -150,
          "centralGravity": 0.01,
          "springLength": 180,
          "springConstant": 0.05,
          "avoidOverlap": 1.0
        }},
        "solver": "forceAtlas2Based",
        "stabilization": {{
          "enabled": true,
          "iterations": 200
        }}
      }},
      "interaction": {{
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "dragNodes": true,
        "tooltipDelay": 150
      }}
    }}
    """
    net.set_options(options_json)
    
    # Save the graph
    net.save_graph(output_filename)
    
    # Custom post-processing of the saved HTML to load Google Font and style tooltips beautifully
    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Inject Google Fonts link
        font_link = '<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
        content = content.replace("<head>", f"<head>\n{font_link}")
        
        # Replace Vis.js CDNs with local files served via static route
        old_css_tag = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" integrity="sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA==" crossorigin="anonymous" referrerpolicy="no-referrer" />'
        old_js_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" integrity="sha512-LnvoEWDFrqGHlHmDD2101OrLcbsfkrzoSpvtSQtxK3RMnRV0eOkhhBN2dXHKRrUU8p2DGRTk35n4O8nWSVe1mQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
        
        content = content.replace(old_css_tag, '<link rel="stylesheet" href="lib/vis-9.1.2/vis-network.css" />')
        content = content.replace(old_js_tag, '<script src="lib/vis-9.1.2/vis-network.min.js"></script>')
        
        # Inject styling for tooltip and scrollbars to fit perfectly
        custom_css = """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
                font-family: 'Outfit', sans-serif !important;
                overflow: hidden;
                width: 100%;
                height: 100%;
            }
            #mynetwork {
                width: 100% !important;
                height: 100% !important;
                background-color: #f8f9fa !important;
            }
            /* Style Vis.js Tooltips to look like modern Google Light popups */
            div.vis-network div.vis-tooltip {
                font-family: 'Outfit', sans-serif !important;
                font-size: 13px !important;
                background-color: #ffffff !important;
                border: 1px solid #dadce0 !important;
                border-radius: 12px !important;
                color: #202124 !important;
                padding: 12px 16px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
                max-width: 320px !important;
                pointer-events: none;
            }
        </style>
        """
        content = content.replace("</head>", f"{custom_css}\n</head>")
        
        # Remove bootstrap card wrapper by replacing body elements to prevent vertical collapse
        content = content.replace('<div class="card" style="width: 100%">', '<div style="width: 100%; height: 100%; margin: 0; padding: 0; border: none;">')
        content = content.replace('<div id="mynetwork" class="card-body"></div>', '<div id="mynetwork" style="width: 100%; height: 100%; margin: 0; padding: 0;"></div>')
        
        # Convert raw string tooltips to DOM elements in Javascript so Vis.js renders them as HTML
        tooltip_js = """
                  // Parse HTML string titles into DOM elements for tooltips
                  nodes.get().forEach(function(node) {
                      if (node.title && typeof node.title === 'string') {
                          var parser = new DOMParser();
                          var doc = parser.parseFromString(node.title, 'text/html');
                          nodes.update({id: node.id, title: doc.body.firstElementChild || doc.body});
                      }
                  });
                  network = new vis.Network(container, data, options);
        """
        content = content.replace("network = new vis.Network(container, data, options);", tooltip_js)
        
        # Remove height overrides in javascript or body
        content = content.replace('height: 600px;', 'height: 100%;')
        content = content.replace('height:500px;', 'height: 100%;')
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(content)
            
    except Exception as e:
        print(f"Error customizing Pyvis HTML file: {e}")
