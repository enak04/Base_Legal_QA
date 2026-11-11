"""
Regression tests for the IPC/CrPC/IEA -> BNS/BNSS/BSA conversion layer.

Most cases come from the masked-intake evaluation (Intake_Eval), where answers
cited penal sections for civil remedies, mapped CrPC procedure numbers to unrelated
BNS offences, or relabelled unmapped IPC numbers as BNS.

These tests need only pandas + torch (no model checkpoints, no OpenAI key).
Run with:  python -m pytest tests/test_bns_converter.py -v
      or:  python -m unittest tests.test_bns_converter -v
"""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from ipc_to_bns.bns_converter import (  # noqa: E402
    BNSConverter,
    CRIMINAL_DOMAIN,
    FAMILY_DOMAIN,
    LABOUR_DOMAIN,
    build_domain_prompt,
    detect_legal_domain,
    extract_statutes_from_text,
    lookup_bnss,
)


class TestCrpcExtraction(unittest.TestCase):
    """Unlabelled procedure sections must not be treated as IPC offences."""

    def test_156_3_before_magistrate_is_crpc(self):
        f = extract_statutes_from_text("I filed an application u/s 156(3) before the CJM")
        self.assertEqual(f["crpcs"], {"156(3)"})
        self.assertEqual(f["ipcs"], set())

    def test_quashing_482_is_crpc(self):
        f = extract_statutes_from_text("I want quashing u/s 482 in the High Court")
        self.assertEqual(f["crpcs"], {"482"})
        self.assertEqual(f["ipcs"], set())

    def test_unlabelled_41a_notice_is_crpc(self):
        f = extract_statutes_from_text("can they send me notice in section 41A")
        self.assertEqual(f["crpcs"], {"41A"})
        self.assertEqual(f["ipcs"], set())

    def test_labelled_crpc(self):
        f = extract_statutes_from_text("notice under section 41A CrPC")
        self.assertEqual(f["crpcs"], {"41A"})
        self.assertEqual(f["domain_sections"], set())

    def test_bail_439_is_crpc(self):
        self.assertEqual(extract_statutes_from_text("bail u/s 439 rejected")["crpcs"], {"439"})

    def test_offences_stay_ipc(self):
        f = extract_statutes_from_text("FIR u/s 420 and 406")
        self.assertEqual(f["ipcs"], {"420", "406"})
        self.assertEqual(f["crpcs"], set())
        self.assertEqual(extract_statutes_from_text("charges 302, 307")["ipcs"], {"302", "307"})

    def test_explicit_ipc_marker(self):
        self.assertEqual(extract_statutes_from_text("booked under Section 498A IPC")["ipcs"], {"498A"})

    def test_fir_for_section_420_is_ipc(self):
        f = extract_statutes_from_text("i am named co accused in an F.I.R for section 420 in karnataka")
        self.assertEqual(f["ipcs"], {"420"})

    def test_section_of_other_act_not_treated_as_ipc(self):
        for t in ("gratuity under section 4 of the Payment of Gratuity Act",
                  "cheque bounce under section 138 of the NI Act",
                  "divorce under section 13B of the Hindu Marriage Act"):
            f = extract_statutes_from_text(t)
            self.assertEqual(f["ipcs"], set(), t)
            self.assertEqual(f["crpcs"], set(), t)


class TestCrpcConcordance(unittest.TestCase):
    """Values checked against the CAPT Bhopal BNSS-CrPC correspondence table."""

    @classmethod
    def setUpClass(cls):
        cls.conv = BNSConverter()

    def test_core_mappings(self):
        expected = {
            "154": ["173"], "156(3)": ["175(3)"], "156": ["175"], "41A": ["35"],
            "125": ["144"], "161": ["180"], "164": ["183"], "167": ["187"], "173": ["193"],
            "437": ["480"], "438": ["482"], "439": ["483"], "482": ["528"],
            "378": ["419"], "468": ["514"],
        }
        for crpc, bnss in expected.items():
            self.assertEqual(lookup_bnss(crpc, self.conv.crpc_to_bnss), bnss, crpc)

    def test_subsection_falls_back_to_base(self):
        self.assertEqual(lookup_bnss("167(2)", self.conv.crpc_to_bnss), ["187"])

    def test_ipc_table_unchanged(self):
        self.assertEqual(self.conv.ipc_to_bns["420"], ["318(4)"])
        self.assertEqual(self.conv.ipc_to_bns["406"], ["316(2)"])
        self.assertEqual(self.conv.ipc_to_bns["494"], ["82(1)"])


class TestDomainDetection(unittest.TestCase):

    def test_divorce_with_cheating_word_is_family(self):
        q = ("I filed divorce on my wife for infidelity. She cheated me. Her lawyer asks 8 lakhs "
             "settlement; no dowry was given. Will court reduce alimony?")
        self.assertEqual(detect_legal_domain(q), FAMILY_DOMAIN)

    def test_maintenance_with_dv_is_family(self):
        q = "My husband left. I want maintenance for me and my daughter. I have evidence of DV complaints."
        self.assertEqual(detect_legal_domain(q), FAMILY_DOMAIN)

    def test_bail_case_mentioning_husband_is_criminal(self):
        q = ("My husband is the 2nd accused and in judicial custody. Bail was rejected in the sessions "
             "court. Police will file the chargesheet.")
        self.assertEqual(detect_legal_domain(q), CRIMINAL_DOMAIN)

    def test_gratuity_is_labour(self):
        q = "The company is releasing me at 4 years 10 months. Their gratuity policy follows the Act."
        self.assertEqual(detect_legal_domain(q), LABOUR_DOMAIN)

    def test_fir_case_is_criminal(self):
        q = "I filed a police complaint, approached the SSP, but no FIR has been registered."
        self.assertEqual(detect_legal_domain(q), CRIMINAL_DOMAIN)


class TestCandidateMenu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conv = BNSConverter()

    def _menu(self, question):
        r = self.conv.process_query(question, [], [], domain_name=detect_legal_domain(question))
        return r["statutory_menu"], r

    def test_alimony_question_gets_no_penal_sections(self):
        menu, r = self._menu(
            "I filed divorce on my wife. She cheated me. No dowry was given. Will court order alimony?")
        self.assertNotIn("BNS (", menu)          # no BNS penal lines
        self.assertNotIn("85", menu)            # cruelty section must not appear for alimony
        self.assertEqual(r["mapped_bns_sections"], [])

    def test_labour_question_has_no_fallback_offences(self):
        menu, _ = self._menu("My employer has not paid gratuity after 4 years 10 months of service.")
        for landmark in ("356(1)", "318(4)", "351(2)", "294", "Defamation"):
            self.assertNotIn(landmark, menu)

    def test_woman_or_provident_fund_words_do_not_add_offences(self):
        menu, _ = self._menu("My sister is a woman employee; her provident fund was not deposited by the employer.")
        self.assertNotIn("316", menu)
        self.assertNotIn("74", menu)

    def test_criminal_question_gets_bns_and_bnss(self):
        menu, r = self._menu("Police registered FIR u/s 420 IPC and threatened me. Anticipatory bail u/s 438?")
        self.assertIn("318(4) BNS", menu)
        self.assertIn("351(1) BNS", menu)
        self.assertIn("Section 482 BNSS (formerly CrPC Section 438)", menu)
        self.assertIn("BNSS 482 (ex-CrPC 438)", r["mapped_bnss_sections"])

    def test_fir_not_registered_points_to_bnss(self):
        menu, _ = self._menu("I filed a police complaint and approached the SSP but police are not registering my FIR.")
        self.assertIn("Section 173 BNSS (formerly CrPC Section 154)", menu)
        self.assertIn("Section 175(3) BNSS (formerly CrPC Section 156(3))", menu)
        self.assertNotIn(" BNS (", menu)

    def test_maintenance_family_gets_bnss_144_but_labour_does_not(self):
        fam, _ = self._menu("My husband left me and our daughter. How do I claim maintenance?")
        self.assertIn("Section 144 BNSS (formerly CrPC Section 125)", fam)
        lab, _ = self._menu("I work in the maintenance department and my salary is unpaid for 3 months.")
        self.assertNotIn("144", lab)

    def _menu_kwargs(self, **over):
        kw = dict(
            explicit_ipcs=set(), kg_ipcs=set(), ipc_to_bns=self.conv.ipc_to_bns,
            explicit_ieas=set(), kg_ieas=set(), iea_to_bsa=self.conv.iea_to_bsa,
            kg_domain_sections={}, modern_semantics=self.conv.modern_semantics,
            question="", is_criminal=True,
        )
        kw.update(over)
        return kw

    def test_menu_capped_at_15_sections(self):
        from ipc_to_bns.bns_converter import build_statutory_candidate_menu
        mapped = sorted(k for k in self.conv.ipc_to_bns if k.isdigit())[:20]   # 20 user-mentioned IPC sections
        menu, mapped_list = build_statutory_candidate_menu(**self._menu_kwargs(explicit_ipcs=set(mapped)))
        lines = [ln for ln in menu.splitlines() if ln.startswith("- Section")]
        self.assertEqual(len(lines), 15)
        self.assertLessEqual(len(mapped_list), 15)

    def test_cap_keeps_user_mentioned_before_graph_and_keywords(self):
        from ipc_to_bns.bns_converter import build_statutory_candidate_menu
        keys = sorted(k for k in self.conv.ipc_to_bns if k.isdigit())
        explicit, graph = set(keys[:10]), set(keys[100:120])
        menu, _ = build_statutory_candidate_menu(**self._menu_kwargs(
            explicit_ipcs=explicit, kg_ipcs=graph,
            question="he threatened me and police won't register my FIR",
            explicit_crpcs={"438"}, crpc_to_bnss=self.conv.crpc_to_bnss, bnss_subjects=self.conv.bnss_subjects,
        ))
        lines = [ln for ln in menu.splitlines() if ln.startswith("- Section")]
        self.assertEqual(len(lines), 15)
        self.assertGreaterEqual(sum("(formerly IPC Section" in ln for ln in lines), 10)   # all user-mentioned IPC kept
        self.assertIn("formerly CrPC Section 438", menu)                                   # user-mentioned CrPC kept
        self.assertIn("formerly CrPC Section 154", menu)                                   # procedure cue kept
        self.assertNotIn("(Graph Discovered)", menu)                                       # lowest priority dropped first

    def test_small_menu_not_affected_by_cap(self):
        menu, _ = self._menu("Police registered FIR u/s 420 IPC and threatened me. Anticipatory bail u/s 438?")
        self.assertIn("318(4) BNS", menu)
        self.assertIn("Section 482 BNSS (formerly CrPC Section 438)", menu)

    def test_semantic_catalogue_search_off_by_default(self):
        import torch
        from ipc_to_bns.bns_converter import build_statutory_candidate_menu
        n = len(self.conv.modern_list)
        kwargs = dict(
            explicit_ipcs=set(), kg_ipcs=set(), ipc_to_bns=self.conv.ipc_to_bns,
            explicit_ieas=set(), kg_ieas=set(), iea_to_bsa=self.conv.iea_to_bsa,
            kg_domain_sections={}, modern_semantics=self.conv.modern_semantics,
            q_emb_norm=torch.ones(1, 4) / 2, modern_embs_norm=torch.ones(n, 4) / 2,
            modern_list=self.conv.modern_list, question="police are not registering my FIR",
            is_criminal=True,
        )
        menu_default, _ = build_statutory_candidate_menu(**kwargs)
        self.assertNotIn("Contextually Retrieved", menu_default)
        self.assertFalse(self.conv.use_semantic_search)
        menu_on, _ = build_statutory_candidate_menu(use_semantic_search=True, **kwargs)
        self.assertIn("Contextually Retrieved", menu_on)

    def test_bsa63_only_with_electronic_evidence(self):
        with_chats, _ = self._menu("Divorce case: will the court accept my wife's WhatsApp chats as proof?")
        without, _ = self._menu("Divorce case: will the court accept my wife's letters as proof?")
        self.assertIn("Section 63 BSA", with_chats)
        self.assertNotIn("63", without)


class TestPrompt(unittest.TestCase):

    def test_civil_prompt_forbids_bns_for_civil_remedies(self):
        p = build_domain_prompt("alimony question", [], [], "", domain_name=FAMILY_DOMAIN)
        self.assertIn("Do NOT cite the Bharatiya Nyaya Sanhita", p)
        self.assertNotIn("MUST NEVER cite any section", p)
        self.assertNotIn("Cruelty: Section 85", p)          # fixed section list removed
        self.assertIn("Section 144 BNSS", p)

    def test_criminal_prompt_separates_bns_bnss_bsa(self):
        p = build_domain_prompt("bail question", [], [], "", domain_name=CRIMINAL_DOMAIN)
        self.assertIn("Bharatiya Nagarik Suraksha Sanhita (BNSS)", p)
        self.assertIn("not the BNS", p)
        self.assertIn("1 July 2024", p)

    def test_bsa63_not_forced(self):
        for dom in (CRIMINAL_DOMAIN, FAMILY_DOMAIN, LABOUR_DOMAIN):
            p = build_domain_prompt("q", [], [], "", domain_name=dom)
            self.assertNotIn("cite Section 63 of BSA, 2023 (formerly Section 65B IEA)", p)
            self.assertIn("only when the person is relying on electronic records", p)


class TestSanitizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conv = BNSConverter()

    def s(self, text):
        return self.conv.sanitize_answer(text)

    def test_ipc_converted_and_old_reference_kept(self):
        out = self.s("Complaint under Section 420 IPC.")
        self.assertIn("Section 318(4) of the Bharatiya Nyaya Sanhita (BNS), 2023", out)
        self.assertIn("(formerly Section 420 IPC)", out)
        self.assertTrue(out.endswith("."))

    def test_crpc_converted_to_bnss(self):
        out = self.s("Seek quashing under Section 482 CrPC.")
        self.assertIn("Section 528 of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023", out)
        self.assertTrue(out.endswith("."))
        out = self.s("Anticipatory bail under Section 438 of the Code of Criminal Procedure.")
        self.assertIn("Section 482 of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023", out)

    def test_unmapped_ipc_not_relabelled_as_bns(self):
        out = self.s("Adultery under Section 497 IPC is no longer an offence.")
        self.assertNotIn("497 of the Bharatiya Nyaya Sanhita", out)
        self.assertIn("repealed IPC", out)

    def test_unmapped_crpc_left_as_written(self):
        self.assertEqual(self.s("Section 999 CrPC applies."), "Section 999 CrPC applies.")

    def test_already_annotated_citation_untouched(self):
        txt = "This is Section 318(4) BNS (formerly Section 420 IPC)."
        self.assertEqual(self.s(txt), txt)

    def test_transition_sentence_untouched(self):
        for txt in ("The Indian Penal Code has been replaced by the BNS.",
                    "Offences before 1 July 2024 remain under the IPC."):
            self.assertEqual(self.s(txt), txt)

    def test_multi_target_mapping(self):
        out = self.s("Cruelty is punishable under Section 498A IPC.")
        self.assertIn("Sections 85 and 86 of the Bharatiya Nyaya Sanhita (BNS), 2023", out)

    def test_iea_converted(self):
        out = self.s("A certificate under Section 65B of the Indian Evidence Act is needed.")
        self.assertIn("Section 63 of the Bharatiya Sakshya Adhiniyam (BSA), 2023", out)
        self.assertIn("formerly Section 65B of the Indian Evidence Act", out)

    def test_no_placeholder_leaks(self):
        texts = ["Section 420 IPC and Section 482 CrPC and Section 497 IPC.",
                 "The Indian Penal Code was replaced; Section 318(4) BNS (formerly Section 420 IPC)."]
        for t in texts:
            out = self.s(t)
            self.assertFalse(any(0xE000 <= ord(ch) <= 0xF8FF for ch in out), out)

    def test_transition_format_requested_by_prompt_untouched(self):
        # build_domain_prompt asks for exactly this format for pre-1 July 2024 matters
        for txt in (
            "You are charged under Section 420 IPC (now Section 318(4) BNS).",
            "Apply under Section 156(3) CrPC (now Section 175(3) BNSS) before the Magistrate.",
            "Under Section 65B of the Evidence Act (now Section 63 BSA), attach a certificate.",
            "Section 498A IPC (now Sections 85 and 86 BNS) applies to the 2022 complaint.",
        ):
            self.assertEqual(self.s(txt), txt)

    def test_plain_citation_next_to_transition_citation_still_converted(self):
        out = self.s("Section 420 IPC (now Section 318(4) BNS) and Section 406 IPC apply.")
        self.assertIn("Section 420 IPC (now Section 318(4) BNS)", out)
        self.assertIn("(formerly Section 406 IPC)", out)


class TestCorpusMismatch(unittest.TestCase):
    """The API serves every question from the Employment & Labour KG / precedent DB."""

    @classmethod
    def setUpClass(cls):
        cls.conv = BNSConverter()
        cls.labour_precedents = [{
            "question": "Lockdown wages not paid",
            "answer": "The employer was booked u/s 188 IPC and section 270 IPC for violating lockdown orders.",
        }]

    def test_criminal_question_ignores_sections_from_labour_precedents(self):
        r = self.conv.process_query(
            "Police refused to register my FIR for cheating by my employer.", [], self.labour_precedents,
            domain_name=CRIMINAL_DOMAIN, corpus_domain=LABOUR_DOMAIN)
        self.assertNotIn("188", r["discovered_ipcs"])
        self.assertNotIn("270", r["discovered_ipcs"])
        self.assertIn("318(4)", r["statutory_menu"])      # keyword-based cheating section kept
        self.assertIn("database and may not match this Criminal Law question", r["prompt"])

    def test_matching_corpus_still_harvests_precedents(self):
        r = self.conv.process_query(
            "Police refused to register my FIR for cheating.", [], self.labour_precedents,
            domain_name=CRIMINAL_DOMAIN, corpus_domain=CRIMINAL_DOMAIN)
        self.assertIn("188", r["discovered_ipcs"])
        self.assertNotIn("may not match", r["prompt"])

    def test_no_corpus_domain_keeps_previous_behaviour(self):
        r = self.conv.process_query(
            "Police refused to register my FIR for cheating.", [], self.labour_precedents,
            domain_name=CRIMINAL_DOMAIN)
        self.assertIn("188", r["discovered_ipcs"])


class TestRequiredFiles(unittest.TestCase):

    def test_required_files_follow_the_given_config(self):
        from inference.config import INFERENCE_CONFIG, get_required_files
        cfg = dict(INFERENCE_CONFIG)
        for key in ("rl_model_path", "dssm_model_path", "retrieval_index",
                    "node_emb_cache_path", "kg_path", "nodes_csv_path"):
            cfg[key] = os.path.join("some_mode_dir", os.path.basename(INFERENCE_CONFIG[key]))
        files = get_required_files(cfg)
        self.assertTrue(all(f["path"].startswith("some_mode_dir") for f in files))
        kg = next(f for f in files if f["path"].endswith("edges.csv"))
        self.assertEqual(kg["name"], "edges.csv")          # name matches the file actually checked
        self.assertEqual(len(get_required_files()), 6)     # default config still works


if __name__ == "__main__":
    unittest.main()
